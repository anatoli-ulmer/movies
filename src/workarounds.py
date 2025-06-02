

def precision_at_k(model, train_matrix, test_matrix, K=10):
    """
    Compute average Precision@K over all users who have at least one item in the test set.
    """
    num_users = train_matrix.shape[0]
    precisions = []

    for user_index in range(num_users):
        test_items = test_matrix[user_index].indices
        if len(test_items) == 0:
            continue

        seen_items = train_matrix[user_index].indices
        recommended = model.recommend(
            user_index,
            train_matrix[user_index],
            N=K + len(seen_items),
            filter_items=seen_items
        )

        # FIXED: unpack item IDs and take top K
        item_ids, _ = recommended
        top_k_items = item_ids[:K]

        hits = len(set(top_k_items) & set(test_items))
        precision = hits / K
        precisions.append(precision)

    return sum(precisions) / len(precisions) if precisions else 0.0


def mean_average_precision_at_k(model, train_matrix, test_matrix, K=10):
    num_users = train_matrix.shape[0]
    average_precisions = []

    for user_index in range(num_users):
        test_items = test_matrix[user_index].indices
        if len(test_items) == 0:
            continue

        seen_items = train_matrix[user_index].indices
        recommended = model.recommend(
            user_index,
            train_matrix[user_index],
            N=K + len(seen_items),
            filter_items=seen_items
        )

        item_ids, _ = recommended
        top_k_items = item_ids[:K]

        hits = 0
        sum_precisions = 0.0

        for i, item_id in enumerate(top_k_items):
            if item_id in test_items:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        if hits > 0:
            average_precisions.append(sum_precisions / hits)
        else:
            average_precisions.append(0.0)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0