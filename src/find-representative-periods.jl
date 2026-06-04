"""
  find_representative_periods(
    clustering_data,
    n_rp;
    drop_incomplete_last_period = false,
    method = :k_means,
    distance = SqEuclidean(),
    initial_representatives = DataFrame(),
    layout = ProfilesTableLayout(),
    kwargs...,
  )

Finds representative periods via data clustering. Honors custom column names via
`layout` (defaults to `(:period, :timestep, :value)`).

Arguments
  - `clustering_data`: long-format data to cluster.
  - `n_rp`: number of representative periods to find.
  - `drop_incomplete_last_period`: controls how the last period is treated if it
    is not complete: if this parameter is set to `true`, the incomplete period
    is dropped and the weights are rescaled accordingly; otherwise, clustering
    is done for `n_rp - 1` periods, and the last period is added as a special
    shorter representative period.
  - `method`: clustering method to use `:k_means`, `:k_medoids`, `:convex_hull`, `:convex_hull_with_null`, or `:conical_hull`.
  - `distance`: semimetric used to measure distance between data points.
  - `initial_representatives`: dataframe of initial RPs. It must use the same key
    columns and follow the same `layout` as `clustering_data`. For hull methods the
    RPs are prepended before clustering; for `:k_means`/`:k_medoids` they are appended
    after clustering.
  - `layout`: `ProfilesTableLayout` describing the column names.
  - other named arguments are forwarded to the clustering method.

# Returns

Returns a `ClusteringResult` with:
  - `profiles::DataFrame`: Long-format representative profiles with columns
    `:rep_period`, `layout.timestep`, all key columns (`auxiliary_data.key_columns`),
    and `layout.value`.
  - `weight_matrix::SparseMatrixCSC{Float64,Int}` (or dense `Matrix{Float64}`):
    rows correspond to source periods and columns to representative periods; entry
    `(p, r)` is the weight of period `p` assigned to representative `r`.
    If the last period is incomplete and `drop_incomplete_last_period` is false,
    it maps to its own representative column with its specific weight; if dropped,
    it is excluded from the rows.
  - `clustering_matrix::Matrix{Float64}`: The feature-by-period matrix used for
    clustering (features are derived from `layout.timestep` crossed with key columns).
  - `rp_matrix::Matrix{Float64}`: The representative profiles in matrix form
    (same feature layout as `clustering_matrix`).
  - `auxiliary_data::AuxiliaryClusteringData`: Auxiliary metadata such as
    `key_columns`, `period_duration`, `last_period_duration`, `n_periods`, and
    (for applicable methods) `medoids` indices.

# Examples

Finding two representatives using default values:
```
julia> df = DataFrame(
           period = kron(1:4, ones(Int, 2)),
           timestep = repeat(1:2, 4),
           profile = "A",
           value = 1:8,
         )

julia> res = TulipaClustering.find_representative_periods(df, 2)
```

Finding two representatives using k-medoids and a custom layout:
```
julia> layout = ProfilesTableLayout(; period = :p, timestep = :ts, value = :val)

julia> df = DataFrame(
           p = kron(1:4, ones(Int, 2)),
           ts = repeat(1:2, 4),
           profile = "A",
           val = 1:8,
         )

julia> res = TulipaClustering.find_representative_periods(df, 2; method = :k_medoids, layout)
```
"""
function find_representative_periods(
    clustering_data::AbstractDataFrame,
    n_rp::Int;
    drop_incomplete_last_period::Bool = false,
    method::Symbol = :k_means,
    distance::SemiMetric = SqEuclidean(),
    initial_representatives::AbstractDataFrame = DataFrame(),
    layout::ProfilesTableLayout = ProfilesTableLayout(),
    kwargs...,
)
    # 1. Check that the number of RPs makes sense. The first check can be done immediately,
    # The second check is done after we compute the auxiliary data
    if n_rp < 1
        throw(
            ArgumentError(
                "The number of representative periods is $n_rp but has to be at least 1.",
            ),
        )
    end

    # Find auxiliary data and pre-compute additional constants that are used multiple times alter
    aux = find_auxiliary_data(clustering_data; layout)
    n_periods = aux.n_periods

    if n_rp > n_periods
        throw(
            ArgumentError(
                "The number of representative periods exceeds the total number of periods, $n_rp > $n_periods.",
            ),
        )
    end

    if drop_incomplete_last_period && n_rp > n_periods - 1
        throw(
            ArgumentError(
                "The number of representative periods exceeds the total number of complete periods when dropping the last incomplete period, $n_rp > $(n_periods - 1).",
            ),
        )
    end

    has_incomplete_last_period = aux.last_period_duration ≠ aux.period_duration
    is_last_period_excluded = has_incomplete_last_period && !drop_incomplete_last_period
    n_complete_periods = has_incomplete_last_period ? n_periods - 1 : n_periods

    # Check that the initial representatives are compatible with the clustering data
    if !isempty(initial_representatives)
        validate_initial_representatives(
            initial_representatives,
            clustering_data,
            aux,
            is_last_period_excluded,
            n_rp,
            layout,
        )
        i_rp = maximum(initial_representatives.period) # number of provided representative periods
    else
        i_rp = 0
    end

    # 2. Find the weights of the two types of periods and pre-build the weight matrix.
    # We assume that the only period that can be incomplete (i.e., has a duration
    # that is less than aux.period_duration) is the very last one. All other periods
    # are complete periods.
    complete_period_weight, incomplete_period_weight = find_period_weights(
        aux.period_duration,
        aux.last_period_duration,
        n_periods,
        drop_incomplete_last_period,
    )

    # In both cases, the weights of the complete periods will be found after clustering.
    if is_last_period_excluded
        weight_matrix = sparse([n_periods], [n_rp], [incomplete_period_weight])
        n_rp -= 1  # incomplete last period becomes its own representative, exclude it from clustering
    else
        weight_matrix = spzeros(n_complete_periods, n_rp)
    end

    # 3. Build the clustering matrix
    clustering_matrix, keys, n_rp = _build_clustering_matrix(
        clustering_data,
        n_rp,
        initial_representatives,
        i_rp,
        method,
        aux,
        n_complete_periods,
        layout,
    )

    # 4. Do the clustering, now that the data is transformed into a matrix
    clustering_matrix, rp_matrix, assignments = _compute_representatives_from_matrix(
        clustering_matrix,
        n_rp,
        initial_representatives,
        i_rp,
        method,
        aux,
        n_complete_periods,
        distance;
        kwargs...,
    )

    # 5. Reinterpret the clustering results into a format we need
    rp_df, weight_matrix, rp_matrix = _reinterpret_clustering_results(
        clustering_data,
        clustering_matrix,
        keys,
        rp_matrix,
        n_rp,
        initial_representatives,
        i_rp,
        method,
        aux,
        n_complete_periods,
        n_periods,
        complete_period_weight,
        weight_matrix,
        is_last_period_excluded,
        distance,
        layout,
    )

    return ClusteringResult(rp_df, weight_matrix, clustering_matrix, rp_matrix, aux)
end

function _build_clustering_matrix(
    clustering_data,
    n_rp,
    initial_representatives,
    i_rp,
    method,
    aux,
    n_complete_periods,
    layout,
)
    period_col = layout.period
    if method in [:k_means, :k_medoids] && !isempty(initial_representatives)
        # If clustering is k-means or k-medoids we remove amount of initial representatives from n_rp
        n_rp -= i_rp
        clustering_matrix, keys = df_to_matrix_and_keys(
            clustering_data[clustering_data[!, period_col] .≤ n_complete_periods, :],
            aux.key_columns;
            layout,
        )

    elseif method in [:convex_hull, :convex_hull_with_null, :conical_hull] &&
           !isempty(initial_representatives)
        # If clustering is one of the hull methods, we add initial representatives to the clustering matrix in front
        updated_clustering_data = deepcopy(clustering_data)
        updated_clustering_data[!, period_col] =
            updated_clustering_data[!, period_col] .+ i_rp
        clustering_data = vcat(initial_representatives, updated_clustering_data)

        clustering_matrix, keys = df_to_matrix_and_keys(
            clustering_data[
                clustering_data[
                    !,
                    period_col,
                ] .≤ (n_complete_periods + maximum(
                    initial_representatives[!, period_col],
                )),
                :,
            ],
            aux.key_columns;
            layout,
        )
    else
        clustering_matrix, keys = df_to_matrix_and_keys(
            clustering_data[clustering_data[!, period_col] .≤ n_complete_periods, :],
            aux.key_columns;
            layout,
        )
    end
    return clustering_matrix, keys, n_rp
end

function _compute_representatives_from_matrix(
    clustering_matrix,
    n_rp,
    initial_representatives,
    i_rp,
    method,
    aux,
    n_complete_periods,
    distance;
    kwargs...,
)
    if n_rp == 0 # If due to the additional representatives we have no clustering, create an empty placeholder
        rp_matrix = nothing
        assignments = Int[]
    elseif method ≡ :k_means
        # Do the clustering
        kmeans_result = kmeans(clustering_matrix, n_rp; distance, kwargs...)

        # Reinterpret the results
        rp_matrix = kmeans_result.centers
        assignments = kmeans_result.assignments
    elseif method ≡ :k_medoids
        # Do the clustering
        # k-medoids uses distance matrix instead of clustering matrix
        distance_matrix = pairwise(distance, clustering_matrix; dims = 2)
        kmedoids_result = kmedoids(distance_matrix, n_rp; kwargs...)

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, kmedoids_result.medoids]
        assignments = kmedoids_result.assignments
        aux.medoids = kmedoids_result.medoids
    elseif method ≡ :convex_hull
        # Do the clustering, with initial indices if provided
        initial_indices = if !isempty(initial_representatives)
            collect(1:i_rp)
        else
            nothing
        end
        hull_indices = greedy_convex_hull(
            clustering_matrix;
            initial_indices = initial_indices,
            n_points = n_rp,
            distance,
            kwargs...,
        )

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, hull_indices]
        assignments = [
            argmin([
                distance(clustering_matrix[:, h], clustering_matrix[:, p + i_rp]) for
                h in hull_indices
            ]) for p in 1:n_complete_periods
        ]
        clustering_matrix = clustering_matrix[:, (i_rp + 1):end]
        aux.medoids = hull_indices
    elseif method ≡ :convex_hull_with_null
        # Check if we can add null to the clustering matrix. The distance to null can
        # be undefined, e.g., for the cosine distance.
        is_distance_to_zero_undefined =
            isnan(distance(zeros(size(clustering_matrix, 1), 1), clustering_matrix[:, 1]))

        if is_distance_to_zero_undefined
            throw(
                ArgumentError(
                    "cannot add null to the clustering data because distance to it is undefined",
                ),
            )
        end

        # Add null to the clustering matrix
        matrix = [zeros(size(clustering_matrix, 1), 1) clustering_matrix]

        # Do the clustering
        hull_indices = greedy_convex_hull(
            matrix;
            n_points = n_rp + 1,
            distance,
            initial_indices = collect(1:(i_rp + 1)),
            kwargs...,
        )

        # Remove null from the beginning and shift all indices by one
        popfirst!(hull_indices)
        hull_indices .-= 1

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, hull_indices]
        assignments = [
            argmin([
                distance(clustering_matrix[:, h], clustering_matrix[:, p + i_rp]) for
                h in hull_indices
            ]) for p in 1:n_complete_periods
        ]
        clustering_matrix = clustering_matrix[:, (i_rp + 1):end]

        aux.medoids = hull_indices
    elseif method ≡ :conical_hull
        # Do a gnomonic projection (normalization) of the data
        normal_vector = vec(mean(clustering_matrix; dims = 2))
        normalize!(normal_vector)
        projection_coefficients = [
            1.0 / dot(normal_vector, clustering_matrix[:, j]) for
            j in axes(clustering_matrix, 2)
        ]
        projected_matrix = [
            clustering_matrix[i, j] * projection_coefficients[j] for
            i in axes(clustering_matrix, 1), j in axes(clustering_matrix, 2)
        ]

        initial_indices = if !isempty(initial_representatives)
            collect(1:i_rp)
        else
            nothing
        end

        hull_indices = greedy_convex_hull(
            projected_matrix;
            n_points = n_rp,
            distance,
            mean_vector = normal_vector,
            initial_indices = initial_indices,
            kwargs...,
        )

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, hull_indices]

        assignments = [
            argmin([
                distance(clustering_matrix[:, h], clustering_matrix[:, p + i_rp]) for
                h in hull_indices
            ]) for p in 1:n_complete_periods
        ]
        clustering_matrix = clustering_matrix[:, (i_rp + 1):end]
    else
        throw(ArgumentError("Clustering method is not supported"))
    end

    return clustering_matrix, rp_matrix, assignments
end

function _reinterpret_clustering_results(
    clustering_data,
    clustering_matrix,
    keys,
    rp_matrix,
    n_rp,
    initial_representatives,
    i_rp,
    method,
    aux,
    n_complete_periods,
    n_periods,
    complete_period_weight,
    weight_matrix,
    is_last_period_excluded,
    distance,
    layout,
)
    period_col = layout.period
    # First, convert the matrix data back to dataframes using the previously saved key columns
    rp_df = if rp_matrix ≡ nothing
        nothing
    else
        matrix_and_keys_to_df(rp_matrix, keys; layout)
    end

    # In case of initial representatives and a non hull method, we add them now
    if !isempty(initial_representatives) && method in [:k_means, :k_medoids]
        representatives_to_add = select!(
            initial_representatives,
            period_col => :rep_period,
            aux.key_columns...,
            layout.value,
        )
        representatives_to_add.rep_period .= representatives_to_add.rep_period .+ n_rp
        rp_df = if rp_df === nothing
            representatives_to_add
        else
            vcat(rp_df, representatives_to_add)
        end
        rename!(rp_df, :rep_period => period_col)
        rp_matrix, keys = df_to_matrix_and_keys(rp_df, aux.key_columns; layout)
        rename!(rp_df, period_col => :rep_period)
        n_rp += i_rp
    end

    # TODO: Verify with Greg if we need this inconditional replacement of assignments or not (it seems like a missing if here)
    assignments = [
        argmin([
            distance(clustering_matrix[:, p], rp_matrix[:, r]) for r in axes(rp_matrix, 2)
        ]) for p in 1:n_complete_periods
    ]

    for (p, rp) in enumerate(assignments)
        weight_matrix[p, rp] = complete_period_weight
    end

    # Next, re-append the last period if it was excluded from clustering
    if is_last_period_excluded
        n_rp += 1
        append_period_from_source_df_as_rp!(
            rp_df;
            source_df = clustering_data,
            period = n_periods,
            rp = n_rp,
            key_columns = aux.key_columns,
            layout = layout,
        )
        if method ≡ :k_medoids
            append!(aux.medoids, n_complete_periods + 1)
        end
    end

    return rp_df, weight_matrix, rp_matrix
end


function build_global_wc_dataframe(profiles, grouped_profiles_data, period_duration, layout)
    ε = 1e-6

    #create the dataframe
    all_rows = DataFrame()

    #Go through every group 
    for (group_idx, (group_key, _)) in enumerate(pairs(grouped_profiles_data))
        #Find the associated rows 
        group_profiles = filter(
            row -> all(row[col] == group_key[col] for col in keys(group_key)),
            profiles
        )

        #Construct the worst case day 
        worst_case_values = Dict{String, Vector{Float64}}()

        for t in 1:period_duration
            timestep_data = filter(row -> row[layout.timestep] == t, group_profiles)
            periods = groupby(timestep_data, :period)

            min_ratio = Inf
            max_demand = 0.0
            min_availability = Inf
            worst_period_index = 0
            for period in periods
                demand_val = only(filter(row -> row.profile_name == "demand", period).value)
                availability_val = sum(filter(row -> row.profile_name in ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"], period).value) #TODO think of other technologies but for tutorial 9 it works 
                ratio = availability_val / (demand_val + ε) # avoid div by zero

                if demand_val > max_demand
                    max_demand = demand_val
                end
                 if ratio < min_ratio
                        min_ratio = ratio
                        worst_period_index = period[1, layout.period]
                end
                    
            end

            worst_row = filter(row -> row[layout.period] == worst_period_index, timestep_data)
            for row in eachrow(worst_row)
                if !haskey(worst_case_values, row.profile_name)
                    worst_case_values[row.profile_name] = Float64[]
                end
                push!(worst_case_values[row.profile_name], row.value)
            end
        end

        # Build rows with period=1 and all groupby key columns
        for (profile_name, values) in worst_case_values
            for (t, v) in enumerate(values)
                row_data = merge(
                    (period=1, timestep=t, profile_name=profile_name, value=v),
                    NamedTuple(col => group_key[col] for col in keys(group_key))
                )
                push!(all_rows, row_data; cols=:union)
            end
        end
    end
    # the columns are expected in this order
    expected = [:period, :timestep, :milestone_year, :scenario, :profile_name, :value]
    all_rows = all_rows[:, expected]
    
    return all_rows
end   

function build_local_before_wc_dataframe(profiles, grouped_profiles_data, pre_results, period_duration, layout)
    ε = 1e-6
    all_rows = DataFrame()

    for (group_key, _) in pairs(grouped_profiles_data)
        group_profiles = filter(
            row -> all(row[col] == group_key[col] for col in keys(group_key)),
            profiles
        )
        clustering_result = pre_results[group_key]
        n_clusters = size(clustering_result.weight_matrix, 2)

        for cluster in 1:n_clusters
            period_indices, _ = findnz(clustering_result.weight_matrix[:, cluster])
            cluster_profiles = filter(row -> row[layout.period] in period_indices, group_profiles)

            worst_case_values = Dict{String, Vector{Float64}}()
            for t in 1:period_duration
                timestep_data = filter(row -> row[layout.timestep] == t, cluster_profiles)
                periods = groupby(timestep_data, :period)

                min_ratio = Inf
                max_demand = 0.0
                min_availability = Inf
                worst_period_index = 0
                for period in periods
                    demand_val = only(filter(row -> row.profile_name == "demand", period).value)
                    availability_val = sum(filter(row -> row.profile_name in ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"], period).value)
                    ratio = availability_val / (demand_val + ε)
                    if demand_val > max_demand
                        max_demand = demand_val
                    end
                     if ratio < min_ratio
                        min_ratio = ratio
                        worst_period_index = period[1, layout.period]
                    end
                end

                worst_row = filter(row -> row[layout.period] == worst_period_index, timestep_data)
                for row in eachrow(worst_row)
                    if !haskey(worst_case_values, row.profile_name)
                        worst_case_values[row.profile_name] = Float64[]
                    end
                    push!(worst_case_values[row.profile_name], row.value)
                end
            end

            for (profile_name, values) in worst_case_values
                for (t, v) in enumerate(values)
                    row_data = merge(
                        (period=cluster, timestep=t, profile_name=profile_name, value=v),
                        NamedTuple(col => group_key[col] for col in keys(group_key))
                    )
                    push!(all_rows, row_data; cols=:union)
                end
            end
        end
    end

    expected = [:period, :timestep, :milestone_year, :scenario, :profile_name, :value]
    return all_rows[:, expected]
end




# function inject_worst_case!(profiles, results_per_group, worst_case, period_duration, distance; layout=ProfilesTableLayout() )
#     # worst_case is :global or :local
#     if (worst_case == :none)
#         println("none")
#         return;
#     end   

#      if worst_case == :global_before || worst_case == :local_before || worst_case == :global_fixed
#         println("$(worst_case) clustering")
#         return
#     end    
    
#     # avoid division by zero
#     ε = 1e-6

#     if (worst_case == :global)
#         println("global")
        
#         for (group_key, clustering_result) in results_per_group  # run for each scenario
#             # println(group_key)
#             # println(keys(group_key))
#             # println(group_key[:milestone_year])
                    
#             # filter profiles to this group
#             group_profiles = filter(row -> all(row[col] == group_key[col] for col in keys(group_key)),  profiles)

#             # Construct the artificial worst case 
#             worst_case_values = Dict{String, Vector{Float64}}()  # profile_name -> 24 values
#             #println(unique(group_profiles.profile_name))

           
#             for t in 1:period_duration
#                 timestep_data = filter(row -> row[layout.timestep] == t, group_profiles);
                
#                 #group each period data together 
#                 periods = groupby(timestep_data, :period);

#                 worst_period_index = 0;
#                 min_ratio = Inf;
#                 max_demand = 0;
#                 min_availability = Inf;
#                 for period in periods
#                     demand_val = only(filter(row -> row.profile_name == "demand", period).value)
#                     availability_val = sum(filter(row -> row.profile_name in ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"], period).value) #TODO think of other technologies but for tutorial 9 it works 
#                     ratio = availability_val / (demand_val + ε) # avoid div by zero

#                     #Update the maximum demand seen and the minimum ratio seen
#                     if demand_val > max_demand
#                         max_demand = demand_val
#                     end

#                     if ratio < min_ratio
#                         min_ratio = ratio
#                         worst_period_index = period[1, layout.period]
#                     end
                    
#                     # Calculate the current score and if it is lower update the worst case timestep
#                     # curr_min_availability = min_ratio * max_demand
#                     # if curr_min_availability < min_availability
#                     #     min_availability = curr_min_availability
#                     #     worst_period_index = period[1, layout.period]
#                     # end
                    
#                 end

#                 # grab the actual values from the worst period at this timestep
#                 worst_row = filter(row -> row[layout.period] == worst_period_index, timestep_data)
#                 for row in eachrow(worst_row)
#                     if !haskey(worst_case_values, row.profile_name)
#                         worst_case_values[row.profile_name] = Float64[]
#                     end
#                     push!(worst_case_values[row.profile_name], row.value)
#                 end       
                
#             end
            
#             # Inject this period into clustering_result struct
#             # First we need to add the worst case rp to the PROFILES 
#             new_rp_index = maximum(clustering_result.profiles.rep_period) + 1
#             new_rows = DataFrame()
#             milestone_year = group_key[:milestone_year]
#             scenario = group_key[:scenario]
#             for (profile_name, values) in worst_case_values
#                 for (t, v) in enumerate(values)
#                     push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, scenario=scenario, profile_name=profile_name, value=v))
#                 end
#             end
#             append!(clustering_result.profiles, new_rows)
            
#             # Append a new column to the WEIGHT_MATRIX for the new global rp
#             # We want to do dirac assignemnt so assign the periods that are closer to the global rp
#             n_original_periods = size(clustering_result.weight_matrix, 1)
#             zero_col = spzeros(n_original_periods)
#             clustering_result.weight_matrix = hcat(clustering_result.weight_matrix, zero_col)
#             # println(size(clustering_result.weight_matrix))
#             # println(clustering_result.weight_matrix[100, :])
#             # for i in 1:n_original_periods
#             #     current_cluster = argmax(clustering_result.weight_matrix[i, :]) #it should have a 1.0 on the column of the cluster it is assigned to 
#             #     period_vec = clustering_result.clustering_matrix[:, i]
#             #     #Get the feature vectors 
#             #     centroid_vec = clustering_result.rp_matrix[:, current_cluster]
#             #     wc_vec = clustering_result.rp_matrix[:, end] # it is the last column 
#             #     # println(current_cluster ===wc_vec )
                
#             #     # compare distances
#             #     if distance(period_vec, wc_vec) <= distance(period_vec, centroid_vec)
#             #         # println(distance(period_vec, wc_vec))
#             #         # println(distance(period_vec, centroid_vec))
#             #         # remove old assignment
#             #         clustering_result.weight_matrix[i, current_cluster] = 0.0
#             #         # assign to new global RP
#             #         clustering_result.weight_matrix[i, end] = 1.0
#             #         #println("CHanged")
#             #     end    
#             # end
#             # # this is for the case that nothing was assinged to the worst case 
#             # # It must have a weight even a small one 
#             # if nnz(clustering_result.weight_matrix[:, end]) == 0 
#             #     println("hello")
#             #     # find the period with the worst stress score globally
#             #     period_scores = [distance(clustering_result.clustering_matrix[:, i], 
#             #                             clustering_result.rp_matrix[:, end]) 
#             #                     for i in 1:n_original_periods]
#             #     worst_p = argmin(period_scores)  # argmin because closer = worse
#             #     clustering_result.weight_matrix[worst_p, argmax(clustering_result.weight_matrix[worst_p, :])] = 0.0
#             #     clustering_result.weight_matrix[worst_p, end] = 1.0
#             #     #println("not found")
#             # end
            
        
#             #Lastly we need to add a column into the RP_MATRIX
#             # build feature vector for worst case
#             wc_df = copy(new_rows)
#             rename!(wc_df, :rep_period => :period)
#             sort!(wc_df, [:profile_name, :timestep])
#             #transform it the format of feature vector 
#             wc_matrix, _ = df_to_matrix_and_keys(wc_df, clustering_result.auxiliary_data.key_columns; layout)
#             clustering_result.rp_matrix = hcat(clustering_result.rp_matrix, wc_matrix)

#             println("Before fitting:")
#             for i in 1:size(clustering_result.weight_matrix, 2)
#                 print("Representative period $i has weight = ")
#                 println(sum(clustering_result.weight_matrix[:, i]))
#             end 
#         end

#     end
    
#     if (worst_case == :local) 
#         println("local")

#         for (group_key, clustering_result) in results_per_group  # run for each scenario
                    
#             clusters = unique(clustering_result.profiles.rep_period)
#             # println(clusters)
#             # Construct the artificial worst case for each cluster
#             for cluster in clusters
#                 # filter profiles to this group and for this cluster 
#                 group_profiles = filter(row -> all(row[col] == group_key[col] for col in keys(group_key)),  profiles)
#                 # get period indices assigned to this cluster from the weight matrix
#                 period_indices, _ = findnz(clustering_result.weight_matrix[:, cluster])
#                 # filter group_profiles to only periods in this cluster
#                 cluster_profiles = filter(row -> row[layout.period] in period_indices, group_profiles)
#                 #Construct the worst case RP
#                 worst_case_values = Dict{String, Vector{Float64}}()  # profile_name -> 24 values
#                 for t in 1:period_duration
#                     timestep_data = filter(row -> row[layout.timestep] == t, cluster_profiles);
                    
#                     #group each period data together 
#                     periods = groupby(timestep_data, :period);

#                     worst_period_index = 0;
#                     min_ratio = Inf;
#                     max_demand = 0;
#                     min_availability = Inf;
#                     for period in periods
#                         demand_val = only(filter(row -> row.profile_name == "demand", period).value)
#                         availability_val = sum(filter(row -> row.profile_name in ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"], period).value)
#                         ratio = availability_val / (demand_val + ε) # avoid div by zero

#                         #Update the maximum demand seen and the minimum ratio seen
#                         if demand_val > max_demand
#                             max_demand = demand_val
#                         end

#                          if ratio < min_ratio
#                             min_ratio = ratio
#                             worst_period_index = period[1, layout.period]
#                         end
                    
                        
#                     end

#                     # grab the actual values from the worst period at this timestep
#                     worst_row = filter(row -> row[layout.period] == worst_period_index, timestep_data)
#                     for row in eachrow(worst_row)
#                         if !haskey(worst_case_values, row.profile_name)
#                             worst_case_values[row.profile_name] = Float64[]
#                         end
#                         push!(worst_case_values[row.profile_name], row.value)
#                     end       
                    
#                 end
                
#                 # Inject this period into clustering_result struct
#                 # First we need to add the worst case rp to the PROFILES
#                 # it gets id +1 from the old maximum which hopefully is already updated from the last iteration of the loop 
#                 new_rp_index = maximum(clustering_result.profiles.rep_period) + 1
#                 new_rows = DataFrame()
#                 milestone_year = group_key[:milestone_year]
#                 scenario = group_key[:scenario]
#                 for (profile_name, values) in worst_case_values
#                     for (t, v) in enumerate(values)
#                         push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, scenario=scenario, profile_name=profile_name, value=v))
#                     end
#                 end
#                 append!(clustering_result.profiles, new_rows)
                
#                 # Append a new column to the WEIGHT_MATRIX with all zeroes for our new global rp 
#                 n_original_periods = size(clustering_result.weight_matrix, 1)
#                 zero_col = spzeros(n_original_periods)
#                 clustering_result.weight_matrix = hcat(clustering_result.weight_matrix, zero_col)
#                 # for i in unique(cluster_profiles.period)
#                 #     current_cluster = argmax(clustering_result.weight_matrix[i, :]) #it should have a 1.0 on the column of the cluster it is assigned to 
#                 #     if(current_cluster > cluster)
#                 #         continue
#                 #     end
#                 #     period_vec = clustering_result.clustering_matrix[:, i]
#                 #     #Get the feature vectors 
#                 #     centroid_vec = clustering_result.rp_matrix[:, current_cluster]
#                 #     wc_vec = clustering_result.rp_matrix[:, end] # it is the last column
                    
#                 #     # compare distances
#                 #     if distance(period_vec, wc_vec) <= distance(period_vec, centroid_vec)
#                 #         # println(distance(period_vec, wc_vec))
#                 #         # println(distance(period_vec, centroid_vec))
#                 #         # remove old assignment
#                 #         clustering_result.weight_matrix[i, current_cluster] = 0.0
#                 #         # assign to new global RP
#                 #         clustering_result.weight_matrix[i, end] = 1.0
#                 #         println("CHanged")
#                 #     end    
#                 # end
#                 # # this is for the case that nothing was assinged to the worst case 
#                 # # It must have a weight even a small one TODO
#                 # #if nnz(clustering_result.weight_matrix[:, end]) == 0 
#                 #     # find the period with the worst stress score globally
#                 #     period_scores = [distance(clustering_result.clustering_matrix[:, i], 
#                 #                             clustering_result.rp_matrix[:, end]) 
#                 #                     for i in period_indices]
#                 #     worst_p = argmin(period_scores)  # argmin because closer = worse
#                 #     clustering_result.weight_matrix[worst_p, argmax(clustering_result.weight_matrix[worst_p, :])] = ε
#                 #     clustering_result.weight_matrix[worst_p, end] = 1.0
#                 #     #println("not found")
#                 # #end

#                 # Lastly we need to add a column into the RP_MATRIX
#                 # build feature vector for worst case
#                 wc_df = copy(new_rows)
#                 rename!(wc_df, :rep_period => :period)
#                 sort!(wc_df, [:profile_name, :timestep])
#                 #transform it the format of feature vector 
#                 wc_matrix, _ = df_to_matrix_and_keys(wc_df, clustering_result.auxiliary_data.key_columns; layout)           
#                 clustering_result.rp_matrix = hcat(clustering_result.rp_matrix, wc_matrix)

#             end
#             println("Before fitting:")
#             for i in 1:size(clustering_result.weight_matrix, 2)
#                 print("Representative period $i has weight = ")
#                 println(sum(clustering_result.weight_matrix[:, i]))
#             end 
#         end
#     end   
# end





#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


function build_global_wc_dataframe_2(profiles, grouped_profiles_data, period_duration, layout)
    ε = 1e-6

    #create the dataframe
    all_rows = DataFrame()

    techs = ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"]

    #Go through every group 
    for (group_idx, (group_key, _)) in enumerate(pairs(grouped_profiles_data))
        #Find the associated rows 
        group_profiles = filter(
            row -> all(row[col] == group_key[col] for col in keys(group_key)),
            profiles
        )

        #Construct the worst case day 
        worst_case_values = Dict{String, Vector{Float64}}()

        for t in 1:period_duration
            timestep_data = filter(row -> row[layout.timestep] == t, group_profiles)

            # Max demand across all periods at this timestep (Kremer: D_r = max_t D_t)
            demand_rows = filter(row -> row.profile_name == "demand", timestep_data)
            max_demand = maximum(demand_rows.value)
            if !haskey(worst_case_values, "demand")
                worst_case_values["demand"] = Float64[]
            end
            push!(worst_case_values["demand"], max_demand)


            # For each technology: min(availability/demand) * max_demand 
            for tech in techs
                tech_rows = filter(row -> row.profile_name == tech, timestep_data)
                isempty(tech_rows) && continue

                min_ratio = Inf
                for p in unique(tech_rows.period)
                    avail = only(filter(row -> row.period == p, tech_rows)).value
                    dem   = only(filter(row -> row.period == p && row.profile_name == "demand", timestep_data)).value
                    ratio = avail / (dem + ε)
                    if ratio < min_ratio
                        min_ratio = ratio
                    end
                end

                min_availability = min_ratio * max_demand
                if !haskey(worst_case_values, tech)
                    worst_case_values[tech] = Float64[]
                end
                push!(worst_case_values[tech], min_availability)
            end
        end

        # Build rows with period=1 and all groupby key columns
        for (profile_name, values) in worst_case_values
            for (t, v) in enumerate(values)
                row_data = merge(
                    (period=1, timestep=t, profile_name=profile_name, value=v),
                    NamedTuple(col => group_key[col] for col in keys(group_key))
                )
                push!(all_rows, row_data; cols=:union)
            end
        end
    end

    # the columns are expected in this order
    expected = [:period, :timestep, :milestone_year, :scenario, :profile_name, :value]
    all_rows = all_rows[:, expected]
    sort!(all_rows, [:period, :profile_name, :timestep])
    return all_rows
end   


function build_local_before_wc_dataframe_2(profiles, grouped_profiles_data, pre_results, period_duration, layout)
    ε = 1e-6
    techs = ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"]
    all_rows = DataFrame()

    for (group_key, _) in pairs(grouped_profiles_data)
        group_profiles = filter(
            row -> all(row[col] == group_key[col] for col in keys(group_key)),
            profiles
        )
        clustering_result = pre_results[group_key]
        n_clusters = size(clustering_result.weight_matrix, 2)

        for cluster in 1:n_clusters
            period_indices, _ = findnz(clustering_result.weight_matrix[:, cluster])
            cluster_profiles = filter(row -> row[layout.period] in period_indices, group_profiles)

            #Construct the worst case day for this cluster
            worst_case_values = Dict{String, Vector{Float64}}()

            for t in 1:period_duration
                timestep_data = filter(row -> row[layout.timestep] == t, cluster_profiles)

                # Max demand across cluster periods at this timestep
                demand_rows = filter(row -> row.profile_name == "demand", timestep_data)
                max_demand = maximum(demand_rows.value)
                if !haskey(worst_case_values, "demand")
                    worst_case_values["demand"] = Float64[]
                end
                push!(worst_case_values["demand"], max_demand)

                # For each technology: min(availability/demand) * max_demand independently
                # Scoped to periods within this cluster only
                for tech in techs
                    tech_rows = filter(row -> row.profile_name == tech, timestep_data)
                    isempty(tech_rows) && continue

                    min_ratio = Inf
                    for p in unique(tech_rows.period)
                        avail = only(filter(row -> row.period == p, tech_rows)).value
                        dem   = only(filter(row -> row.period == p && row.profile_name == "demand", timestep_data)).value
                        ratio = avail / (dem + ε)
                        if ratio < min_ratio
                            min_ratio = ratio
                        end
                    end

                    min_availability = min_ratio * max_demand
                    if !haskey(worst_case_values, tech)
                        worst_case_values[tech] = Float64[]
                    end
                    push!(worst_case_values[tech], min_availability)
                end
            end

            # Build rows with period=cluster so each cluster WC gets a unique period index
            for (profile_name, values) in worst_case_values
                for (t, v) in enumerate(values)
                    row_data = merge(
                        (period=cluster, timestep=t, profile_name=profile_name, value=v),
                        NamedTuple(col => group_key[col] for col in keys(group_key))
                    )
                    push!(all_rows, row_data; cols=:union)
                end
            end
        end
    end

    expected = [:period, :timestep, :milestone_year, :scenario, :profile_name, :value]
    all_rows = all_rows[:, expected]
    sort!(all_rows, [:period, :profile_name, :timestep])

    return all_rows
end


function inject_worst_case_2!(profiles, results_per_group, worst_case, weight_type, period_duration, distance; layout=ProfilesTableLayout())
    if worst_case == :none
        println("none")
        return
    end
    if worst_case == :global_before || worst_case == :local_before || worst_case == :global_fixed
        println("$(worst_case) clustering")
        return
    end

    # avoid division by zero
    ε = 1e-6

    techs = ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"]

    if worst_case == :global
        println("global")
        for (group_key, clustering_result) in results_per_group # run for each scenario
            
            # filter profiles to this group
            group_profiles = filter(
                row -> all(row[col] == group_key[col] for col in keys(group_key)),
                profiles
            )

            # Construct the artificial worst case 
            worst_case_values = Dict{String, Vector{Float64}}()  # profile_name -> 24 values
            #println(unique(group_profiles.profile_name))

            for t in 1:period_duration
                timestep_data = filter(row -> row[layout.timestep] == t, group_profiles)

                # Max demand across all periods at this timestep
                demand_rows = filter(row -> row.profile_name == "demand", timestep_data)
                max_demand = maximum(demand_rows.value)

                if !haskey(worst_case_values, "demand")
                    worst_case_values["demand"] = Float64[]
                end
                push!(worst_case_values["demand"], max_demand)

                # For each technology find the availability by: min(availability/demand) * max_demand
                for tech in techs
                    
                    tech_rows = filter(row -> row.profile_name == tech, timestep_data)
                    isempty(tech_rows) && continue

                    min_ratio = Inf
                    for period in unique(tech_rows.period)
                        avail = only(filter(row -> row.period == period, tech_rows)).value
                        dem   = only(filter(row -> row.period == period && row.profile_name == "demand", timestep_data)).value
                        ratio = avail / (dem + ε)
                        if ratio < min_ratio
                            min_ratio = ratio
                        end
                    end
                    
                    min_availability = min_ratio * max_demand
                    if !haskey(worst_case_values, tech)
                        worst_case_values[tech] = Float64[]
                    end
                    push!(worst_case_values[tech], min_availability)
                end
            end
            # Inject this period into clustering_result struct
            # First we need to add the worst case rp to the PROFILES 
            new_rp_index = maximum(clustering_result.profiles.rep_period) + 1
            new_rows = DataFrame()
            milestone_year = group_key[:milestone_year]
            has_scenario = :scenario in keys(group_key)
            for (profile_name, values) in worst_case_values
                for (t, v) in enumerate(values)
                    if has_scenario
                        push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, scenario=group_key[:scenario], profile_name=profile_name, value=v))
                    else
                        push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, profile_name=profile_name, value=v))
                    end
                end
            end
            append!(clustering_result.profiles, new_rows)
            
            # Append a new column to the WEIGHT_MATRIX for the new global rp
            # We want to do dirac assignemnt so assign the periods that are closer to the global rp
            n_original_periods = size(clustering_result.weight_matrix, 1)
            zero_col = spzeros(n_original_periods)
            clustering_result.weight_matrix = hcat(clustering_result.weight_matrix, zero_col) 
            if weight_type == :dirac
                # Full Dirac reassignment: for each original period, if it is closer
                # to the global WC than to its currently assigned centroid, steal it.
                wc_vec = clustering_result.rp_matrix[:, end]
                n_reassigned = 0
                for i in 1:n_original_periods
                    period_vec  = clustering_result.clustering_matrix[:, i]
                    current_col = argmax(Vector(clustering_result.weight_matrix[i, 1:end-1])) # Last column is the global
                    centroid_vec = clustering_result.rp_matrix[:, current_col]
                    dist_to_centroid = distance(period_vec, centroid_vec)
                    dist_to_wc       = distance(period_vec, wc_vec)
                    if dist_to_wc < dist_to_centroid
                        clustering_result.weight_matrix[i, current_col] = 0.0
                        clustering_result.weight_matrix[i, end]         = 1.0
                        n_reassigned += 1
                    end
                end
                println("  [global WC] Dirac reassignment: $n_reassigned / $n_original_periods periods moved to WC")
            end
           
        
            #Lastly we need to add a column into the RP_MATRIX
            # build feature vector for worst case
            wc_df = copy(new_rows)
            rename!(wc_df, :rep_period => :period)
            sort!(wc_df, [:profile_name, :timestep])
            #transform it the format of feature vector 
            wc_matrix, _ = df_to_matrix_and_keys(wc_df, clustering_result.auxiliary_data.key_columns; layout)
            clustering_result.rp_matrix = hcat(clustering_result.rp_matrix, wc_matrix)

            # println("Before fitting:")
            # for i in 1:size(clustering_result.weight_matrix, 2)
            #     print("Representative period $i has weight = ")
            #     println(sum(clustering_result.weight_matrix[:, i]))
            # end 
        end
    end

    if worst_case == :local
        println("local")

        for (group_key, clustering_result) in results_per_group
            # filter profiles to this group and for this cluster 
            group_profiles = filter(row -> all(row[col] == group_key[col] for col in keys(group_key)),  profiles)
            clusters = unique(clustering_result.profiles.rep_period)
            # Construct the artificial worst case for each cluster
            for cluster in clusters
                
                # get period indices assigned to this cluster from the weight matrix
                period_indices, _ = findnz(clustering_result.weight_matrix[:, cluster])
                # filter group_profiles to only periods in this cluster
                cluster_profiles = filter(row -> row[layout.period] in period_indices, group_profiles)
                #Construct the worst case RP

                worst_case_values = Dict{String, Vector{Float64}}() # profile_name -> 24 values

                for t in 1:period_duration
                    timestep_data = filter(row -> row[layout.timestep] == t, cluster_profiles)

                    # Max demand across cluster periods at this timestep
                    demand_rows = filter(row -> row.profile_name == "demand", timestep_data)
                    max_demand = maximum(demand_rows.value)
                    if !haskey(worst_case_values, "demand")
                        worst_case_values["demand"] = Float64[]
                    end
                    push!(worst_case_values["demand"], max_demand)

                    # For each technology find the availability by: min(availability/demand) * max_demand
                    for tech in techs
                        tech_rows = filter(row -> row.profile_name == tech, timestep_data)
                        isempty(tech_rows) && continue

                        min_ratio = Inf
                        for p in unique(tech_rows.period)
                            avail = only(filter(row -> row.period == p, tech_rows)).value
                            dem   = only(filter(row -> row.period == p && row.profile_name == "demand", timestep_data)).value
                            ratio = avail / (dem + ε)
                            if ratio < min_ratio
                                min_ratio = ratio
                            end
                        end

                        min_availability = min_ratio * max_demand
                        if !haskey(worst_case_values, tech)
                            worst_case_values[tech] = Float64[]
                        end
                        push!(worst_case_values[tech], min_availability)
                    end
                end
                # Inject this period into clustering_result struct
                # First we need to add the worst case rp to the PROFILES
                # it gets id +1 from the old maximum which hopefully is already updated from the last iteration of the loop 
                new_rp_index = maximum(clustering_result.profiles.rep_period) + 1
                new_rows = DataFrame()
                milestone_year = group_key[:milestone_year]
                has_scenario = :scenario in keys(group_key)
                for (profile_name, values) in worst_case_values
                    for (t, v) in enumerate(values)
                        if has_scenario
                            push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, scenario=group_key[:scenario], profile_name=profile_name, value=v))
                        else
                            push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, profile_name=profile_name, value=v))
                        end
                    end
                end
                append!(clustering_result.profiles, new_rows)

                # Append a new column to the WEIGHT_MATRIX with all zeroes for our new global rp 
                n_original_periods = size(clustering_result.weight_matrix, 1)
                zero_col = spzeros(n_original_periods)
                clustering_result.weight_matrix = hcat(clustering_result.weight_matrix, zero_col)
                if weight_type == :dirac
                    # Full Dirac reassignment scoped to this cluster's periods only.
                    # For each period in the cluster, if it is closer to the local WC
                    # than to its current centroid, steal it.
                    wc_vec       = clustering_result.rp_matrix[:, end]
                    n_reassigned = 0
                    for i in period_indices
                        period_vec   = clustering_result.clustering_matrix[:, i]
                        current_col  = argmax(Vector(clustering_result.weight_matrix[i, 1:end-1])) # the last added local worst case is the last column 
                        centroid_vec = clustering_result.rp_matrix[:, current_col]
                        dist_to_centroid = distance(period_vec, centroid_vec)
                        dist_to_wc       = distance(period_vec, wc_vec)
                        if dist_to_wc < dist_to_centroid
                            clustering_result.weight_matrix[i, current_col] = 0.0
                            clustering_result.weight_matrix[i, end]         = 1.0
                            n_reassigned += 1
                        end
                    end
                    println("  [local WC cluster=$cluster] Dirac reassignment: $n_reassigned / $(length(period_indices)) periods moved to WC")
                end


                # Lastly we need to add a column into the RP_MATRIX
                # build feature vector for worst case
                wc_df = copy(new_rows)
                rename!(wc_df, :rep_period => :period)
                sort!(wc_df, [:profile_name, :timestep])
                #transform it the format of feature vector 
                wc_matrix, _ = df_to_matrix_and_keys(wc_df, clustering_result.auxiliary_data.key_columns; layout)           
                clustering_result.rp_matrix = hcat(clustering_result.rp_matrix, wc_matrix)
            end

            # println("Before fitting:")
            # for i in 1:size(clustering_result.weight_matrix, 2)
            #     println("Representative period $i has weight = $(sum(clustering_result.weight_matrix[:, i]))")
            # end
        end
    end
end



function inject_worst_case_fixed_2!(profiles, results_per_group, worst_case, period_duration, distance, percentage; layout=ProfilesTableLayout())
    if worst_case != :global_fixed
        return
    end

    # avoid division by zero
    ε = 1e-6

    techs = ["wind_onshore", "wind_offshore", "solar", "hydro_inflow"]

    if worst_case == :global_fixed
        println("global_fixed")
        for (group_key, clustering_result) in results_per_group # run for each scenario
             
            # println("Before fitting:")
            # for i in 1:size(clustering_result.weight_matrix, 2)
            #     print("Representative period $i has weight = ")
            #     println(sum(clustering_result.weight_matrix[:, i]))
            # end 
            # filter profiles to this group
            group_profiles = filter(
                row -> all(row[col] == group_key[col] for col in keys(group_key)),
                profiles
            )

            # Construct the artificial worst case 
            worst_case_values = Dict{String, Vector{Float64}}()  # profile_name -> 24 values
            #println(unique(group_profiles.profile_name))

            for t in 1:period_duration
                timestep_data = filter(row -> row[layout.timestep] == t, group_profiles)

                # Max demand across all periods at this timestep
                demand_rows = filter(row -> row.profile_name == "demand", timestep_data)
                max_demand = maximum(demand_rows.value)

                if !haskey(worst_case_values, "demand")
                    worst_case_values["demand"] = Float64[]
                end
                push!(worst_case_values["demand"], max_demand)

                # For each technology find the availability by: min(availability/demand) * max_demand
                for tech in techs
                    tech_rows = filter(row -> row.profile_name == tech, timestep_data)
                    isempty(tech_rows) && continue

                    min_ratio = Inf
                    for period in unique(tech_rows.period)
                        avail = only(filter(row -> row.period == period, tech_rows)).value
                        dem   = only(filter(row -> row.period == period && row.profile_name == "demand", timestep_data)).value
                        ratio = avail / (dem + ε)
                        if ratio < min_ratio
                            min_ratio = ratio
                        end
                    end

                    min_availability = min_ratio * max_demand
                    if !haskey(worst_case_values, tech)
                        worst_case_values[tech] = Float64[]
                    end
                    push!(worst_case_values[tech], min_availability)
                end
            end
            # Inject this period into clustering_result struct
            # First we need to add the worst case rp to the PROFILES 
            new_rp_index = maximum(clustering_result.profiles.rep_period) + 1
            new_rows = DataFrame()
            milestone_year = group_key[:milestone_year]
            has_scenario = :scenario in keys(group_key)
            for (profile_name, values) in worst_case_values
                for (t, v) in enumerate(values)
                    if has_scenario
                        push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, scenario=group_key[:scenario], profile_name=profile_name, value=v))
                    else
                        push!(new_rows, (rep_period=new_rp_index, timestep=t, milestone_year=milestone_year, profile_name=profile_name, value=v))
                    end
                end
            end

            append!(clustering_result.profiles, new_rows)
            
            # Append a new column to the WEIGHT_MATRIX for the new global rp
            # We want to assing to it 10%
            n_original_periods = size(clustering_result.weight_matrix, 1)
            total_weight       = sum(clustering_result.weight_matrix)
            wc_weight          = percentage * total_weight
            scale_factor       = (total_weight - wc_weight) / total_weight

            clustering_result.weight_matrix .*= scale_factor

            wc_col = spzeros(n_original_periods)
            wc_col[end] = wc_weight
            clustering_result.weight_matrix = hcat(clustering_result.weight_matrix, wc_col)
        
            #Lastly we need to add a column into the RP_MATRIX
            # build feature vector for worst case
            wc_df = copy(new_rows)
            rename!(wc_df, :rep_period => :period)
            sort!(wc_df, [:profile_name, :timestep])
            #transform it the format of feature vector 
            wc_matrix, _ = df_to_matrix_and_keys(wc_df, clustering_result.auxiliary_data.key_columns; layout)
            clustering_result.rp_matrix = hcat(clustering_result.rp_matrix, wc_matrix)
        end
    end
end