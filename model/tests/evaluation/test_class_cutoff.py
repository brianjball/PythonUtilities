from unittest import TestCase
from random import shuffle
from model_utilities.evaluation.class_cutoff import find_cutoff_sorted, find_cutoff, find_multiclass_cutoffs
from model_utilities.evaluation.cutoff_methods import CutoffMethod


class TestClassCutoff(TestCase):
    def setUp(self):
        self.data = [(0, 0),
                     (0, 0.04),
                     (0, 0.08),
                     (0, 0.12),
                     (0, 0.16),
                     (0, 0.2),
                     (0, 0.24),
                     (1, 0.28),
                     (0, 0.32),
                     (0, 0.36),
                     (1, 0.4),
                     (0, 0.44),
                     (1, 0.48),
                     (1, 0.52),
                     (1, 0.56),
                     (0, 0.6),
                     (0, 0.64),
                     (0, 0.68),
                     (1, 0.72),
                     (1, 0.76),
                     (1, 0.8),
                     (1, 0.84),
                     (1, 0.88),
                     (1, 0.92),
                     (1, 0.96),
                     (1, 1)]
        self.data.reverse()

    def test_find_cutoff_sorted(self):
        cutoff_sorted = find_cutoff_sorted(self.data, CutoffMethod.AbsoluteDistance, 1)
        self.assertLess(0.5, cutoff_sorted)

    def test_find_cutoff(self):
        randomized_list = [x for x in self.data]
        shuffle(randomized_list)

        split_list = list(zip(*self.data))
        split_randomized_list = list(zip(*randomized_list))

        self.assertEqual(find_cutoff_sorted(self.data, CutoffMethod.AbsoluteDistance, 1),
                         find_cutoff(split_list[0], split_list[1], CutoffMethod.AbsoluteDistance, 1))
        self.assertEqual(find_cutoff(split_list[0], split_list[1], CutoffMethod.AbsoluteDistance, 1),
                         find_cutoff(split_randomized_list[0], split_randomized_list[1], CutoffMethod.AbsoluteDistance, 1))

    def test_find_multiclass_cutoff_is_symmetric_for_two_class(self):
        split_list = list(zip(*self.data))
        actual = ["b" if x == 1 else "a" for x in split_list[0]]
        predicted = [(1-x, x) for x in split_list[1]]
        response = find_multiclass_cutoffs(actual, predicted, {"a": 0, "b": 1}, CutoffMethod.AbsoluteDistance, 1)
        print(response)
        self.assertLessEqual(response["a"], response["b"])  # These may not exactly match, but should be close.
