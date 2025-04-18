import unittest
from src.matching_score import rank_resumes_by_similarity
import numpy as np

class TestResumeMatching(unittest.TestCase):

    def test_rank_resumes_by_similarity_returns_sorted_results(self):
        job_description = "Looking for a data scientist skilled in machine learning and Python."
        resumes = [
            "Experienced data scientist with strong Python and ML skills.",
            "Marketing expert with SEO and content creation experience.",
            "Software engineer familiar with Java and backend development."
        ]

        result = rank_resumes_by_similarity(job_description, resumes)

        # Assert result is a list of tuples (index, score)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], int)
            self.assertIsInstance(item[1], (float, np.floating)) 

        # Assert the result is sorted in descending order by similarity score
        scores = [score for _, score in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_resumes_by_similarity_empty_input(self):
        with self.assertRaises(ValueError):
            rank_resumes_by_similarity("", [])


