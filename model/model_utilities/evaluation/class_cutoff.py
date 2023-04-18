# FIXME! Need to be able to distinguish counts versus rates.
from typing import Sequence, Tuple, Dict, Any

from model_utilities.evaluation.cutoff_methods import CutoffMethod


def _adjust_score(current_score: float,
                  false_negatives: float,
                  false_positives: float,
                  new_true_positives: float,
                  new_false_positives: float,
                  cutoff_method: CutoffMethod,
                  false_positive_versus_negative_importance: float,
                  denominator_positives: float,
                  denominator_negatives: float
                  ) -> float:
    if cutoff_method == CutoffMethod.AbsoluteDistance:
        return current_score - new_true_positives / denominator_positives + new_false_positives * false_positive_versus_negative_importance / denominator_negatives
    elif cutoff_method == CutoffMethod.DistanceSquared:
        effective_positive_denominator = denominator_positives * denominator_positives
        effective_negative_denominator = denominator_negatives * denominator_negatives

        score_adjustment = (new_true_positives * new_true_positives - 2 * new_true_positives * false_negatives) / effective_positive_denominator
        score_adjustment += (false_positive_versus_negative_importance * (new_false_positives * new_false_positives + 2 * false_positives * new_false_positives)) / effective_negative_denominator
        return current_score + score_adjustment


def find_cutoff_sorted(sorted_test_pairs: Sequence[Tuple[float, float]],
                       cutoff_method: CutoffMethod,
                       false_positive_versus_negative_importance: float,
                       use_rates_over_counts: bool = True) -> float:
    """
    Finds the optimal cutoff for a binary classification when provided a ratio of how important false positives are versus false negatives and a distance metric. Provide the test set tu run this test.

    This algorithm works by taking advantage of the fact that you can easily calculate the change in overall score when moving the cutoff just enough to shift a single point. The trick is that the confusion matrix will only move in one of two ways: a true negative becomes a false positive or a false negative becomes a true positive. The starting condition is that all the points are true and false negatives and will end as true and false positives. Thus it can iteratively minimize the penalty in O(n) time.

    Args:
        sorted_test_pairs: List of pairs of observed value and its predicted probability according to your model_utilities, sorted in descending order of probability.
        cutoff_method: Cutoff metric you would like applied.
        false_positive_versus_negative_importance: Ratio of the penalties associated to false positives over false negatives.
        use_rates_over_counts: Whether to use rates (True) or counts (False) when evaluating the penalties.

    Returns:
        The cutoff probability that optimizes the balance of false positives and false negatives.
    """
    n = len(sorted_test_pairs)
    false_negatives = sum((x[0] for x in sorted_test_pairs))
    false_positives = 0
    denominator_positives = false_negatives if use_rates_over_counts else 1
    denominator_negatives = n - denominator_positives if use_rates_over_counts else 1

    # We can't write a straight for loop here because multiple points may have the same probability, e.g. in a tree model_utilities.
    #  This algorithm's speed depends on the specific implementation of the iterable type passed in (array vs linked list vs vector). We're omitting this detail right now.
    i = 0
    if cutoff_method == CutoffMethod.AbsoluteDistance:
        current_score = false_negatives / denominator_positives
    elif cutoff_method == CutoffMethod.DistanceSquared:
        current_score = (false_negatives * false_negatives) / (denominator_positives * denominator_positives)
    best_cutoff = i
    best_score = current_score

    while i < n:
        j = 0
        new_true_positives = 0
        new_false_positives = 0
        while i+j < n and sorted_test_pairs[i][1] == sorted_test_pairs[i+j][1]:
            if sorted_test_pairs[i+j][0]:
                new_true_positives += 1
            else:
                new_false_positives += 1
            j += 1
        current_score = _adjust_score(current_score,
                                      false_negatives,
                                      false_positives,
                                      new_true_positives,
                                      new_false_positives,
                                      cutoff_method,
                                      false_positive_versus_negative_importance,
                                      denominator_positives,
                                      denominator_negatives)

        if current_score < best_score:
            best_score = current_score
            best_cutoff = sorted_test_pairs[i][1]

        i += j
        if j == 0:
            break

    return best_cutoff


def find_cutoff(actual: Sequence[float],
                predicted: Sequence[float],
                cutoff_method: CutoffMethod = CutoffMethod.AbsoluteDistance,
                false_positive_versus_negative_importance: float = 1,
                use_rates_over_counts: bool = True) -> float:
    """
    Finds the optimal cutoff for a binary classification when provided a ratio of how important false positives are versus false negatives and a distance metric. Provide the test set for this test.

    Args:
        actual: List of observed values (true positive is 1, true negative is 0).
        predicted: List of probabilities of a positive value. Must match the ordering of the actuals provided.
        cutoff_method: Cutoff metric you would like applied.
        false_positive_versus_negative_importance: Ratio of the penalties associated to false positives over false negatives.
        use_rates_over_counts: Whether to use rates (True) or counts (False) when evaluating the penalties.

    Returns:
        The cutoff probability that optimizes the balance of false positives and false negatives.
    """
    sorted_test_pairs = sorted(zip(actual, predicted), key=lambda x: x[1], reverse=True)
    return find_cutoff_sorted(sorted_test_pairs, cutoff_method, false_positive_versus_negative_importance, use_rates_over_counts)


def find_multiclass_cutoffs(actual: Sequence[Any],
                            predicted: Sequence[Sequence[float]],
                            class_map: Dict[Any, int],
                            cutoff_method: CutoffMethod = CutoffMethod.AbsoluteDistance,
                            false_positive_versus_negative_importance: float = 1,
                            use_rates_over_counts: bool = True) -> Dict[Any, float]:
    """
    Finds the optimal cutoffs for a multiclass classification model_utilities. As each class is handled independently, this may produce slightly counterintuitive results. First, you may have a prediction that according this is in multiple classes - this is technically fine but may provide unsatisfying results. You will require an additional test to determine between the multiple resulting classes. It is also possible that this will tell you that a prediction doesn't belong to ANY class - this is valuable information and you should look into that area of parameter space specifically to understand why the model_utilities is confused there.

    Note that in binary classification, this produces redundant information since one class is the opposite of the other. In practice, the values may deviate a little if the exact value of the cutoff makes no difference to the final score.

    Args:
        actual: List of observed values.
        predicted: List of probabilities for the various classes. Must match the ordering of the actuals provided.
        class_map: Dictionary of class type to the order of the classes in the inner prediction lists.
        cutoff_method: Cutoff metric you would like applied.
        false_positive_versus_negative_importance: Ratio of the penalties associated to false positives over false negatives.
        use_rates_over_counts: Whether to use rates (True) or counts (False) when evaluating the penalties.

    Returns:
        Dictionary of the classes and their respective cutoffs.
    """
    results = {this_class: find_cutoff([1 if this_actual == this_class else 0 for this_actual in actual],
                                       [this_predicted[index] for this_predicted in predicted],
                                       cutoff_method,
                                       false_positive_versus_negative_importance,
                                       use_rates_over_counts
                                       ) for this_class, index in class_map.items()}
    return results
