import unittest
import pandas as pd
from src.extract_skills_edu_exp import extract_skill_edu_exp  

class TestResumeParser(unittest.TestCase):

    def setUp(self):
        self.sample_resume = """
        John Doe
        Software Engineer with 5 years of experience in Python, Django, and REST APIs.
        Skilled in cloud platforms like AWS and Azure.
        Worked at ABC Corporation as a Backend Developer.
        Education: Bachelor of Technology in Computer Science from XYZ University.
        Certified in AWS Cloud Practitioner.
        """

    def test_extract_skill_edu_exp_fields(self):
        result = extract_skill_edu_exp(self.sample_resume)

        self.assertIsInstance(result, pd.Series)
        self.assertIn("Skills", result)
        self.assertIn("Education", result)
        self.assertIn("Experience", result)

        # Assert that the result contains expected keywords
        self.assertIn("AWS", result["Skills"])
        self.assertIn("Bachelor of Technology", result["Education"])
        self.assertIn("Worked at ABC Corporation", result["Experience"])

    def test_extract_skill_edu_exp_empty_text(self):
        empty_resume = ""
        result = extract_skill_edu_exp(empty_resume)

        self.assertEqual(result["Skills"], "")
        self.assertEqual(result["Education"], "")
        self.assertEqual(result["Experience"], "")


