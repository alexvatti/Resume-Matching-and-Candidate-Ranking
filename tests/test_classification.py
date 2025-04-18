import unittest
from src.classification import predict_resume_category
import pandas as pd

class TestResumeClassification(unittest.TestCase):

    def test_predict_resume_category(self):
        # Test Case 1: Valid Data Science Resume
        sample_resume_1 = "Expert in data science and machine learning with experience in pandas and data visualization."
        result_1 = predict_resume_category(sample_resume_1)
        self.assertEqual(result_1, ["Data Science"])

        # Test Case 2: Resume with Conflicting Skills
        sample_resume_2 = "Proficient in Java development, experienced in data pipelines."
        result_2 = predict_resume_category(sample_resume_2)
        self.assertIn(result_2, [["Java Developer"], ["Data Science"]])

        # Test Case 3: Unrelated Content
        sample_resume_3 = "I love cooking and traveling."
        result_3 = predict_resume_category(sample_resume_3)
        self.assertEqual(result_3, ["Health and fitness"])

        # Test Case 4: Minimal Resume
        sample_resume_4 = "SAP and FICO expertise."
        result_4 = predict_resume_category(sample_resume_4)
        self.assertEqual(result_4, ["SAP Developer"])

        # Test Case 5: Health and Fitness Domain
        sample_resume_5 = "Dietician with expertise in fitness plans and workout routines."
        result_5 = predict_resume_category(sample_resume_5)
        self.assertEqual(result_5, ["Health and fitness"])

        # Test Case 6: Resume with Multiple Technologies
        sample_resume_6 = "Proficient in Python, Java, React, and Machine Learning. Experience in DevOps."
        result_6 = predict_resume_category(sample_resume_6)
        self.assertIn(result_6, [["Python Developer"], ["DevOps Engineer"]])

        # Test Case 7: Network Security Engineer
        sample_resume_7 = "Expert in firewalls, cybersecurity, and intrusion detection systems."
        result_7 = predict_resume_category(sample_resume_7)
        self.assertEqual(result_7, ["Network Security Engineer"])

        # Test Case 8: Automation Testing
        sample_resume_8 = "Experienced with Selenium, test automation, Jenkins, and test case development."
        result_8 = predict_resume_category(sample_resume_8)
        self.assertEqual(result_8, ["Automation Testing"])

        # Test Case 9: PMO Domain
        sample_resume_9 = "Project management office experience with managing timelines, milestones, and deliverables."
        result_9 = predict_resume_category(sample_resume_9)
        self.assertEqual(result_9, ["PMO"])

        # Test Case 10: Electrical Engineering
        sample_resume_10 = "Strong background in circuits, voltage systems, transformers, and power distribution."
        result_10 = predict_resume_category(sample_resume_10)
        self.assertIn(result_10, [["Testing"],["Electrical Engineering"]])

    def test_predict_job_description_category(self):
        # Test Case 1: Valid Data Science JD
        sample_jd_1 = "Looking for an expert in data science, machine learning, and pandas for data analysis."
        result_1 = predict_resume_category(sample_jd_1)
        self.assertEqual(result_1, ["Data Science"])

        # Test Case 2: JD with Conflicting Skills
        sample_jd_2 = "Seeking a developer with expertise in Java and experience with data pipelines."
        result_2 = predict_resume_category(sample_jd_2)
        self.assertIn(result_2, [["Java Developer"], ["Data Science"]])

        # Test Case 3: Unrelated JD
        sample_jd_3 = "We are hiring for a role in marketing with a focus on brand management and advertising."
        result_3 = predict_resume_category(sample_jd_3)
        self.assertEqual(result_3, ["Sales"])

        # Test Case 4: Minimal JD
        sample_jd_4 = "Looking for an SAP developer with FICO experience."
        result_4 = predict_resume_category(sample_jd_4)
        self.assertEqual(result_4, ["SAP Developer"])

        # Test Case 5: Health and Fitness JD
        sample_jd_5 = "Looking for a dietician or fitness coach to develop workout plans and diet strategies."
        result_5 = predict_resume_category(sample_jd_5)
        self.assertEqual(result_5, ["Health and fitness"])

        # Test Case 6: JD with Multiple Technologies
        sample_jd_6 = "We need a Python developer, experienced in Java, React, and DevOps."
        result_6 = predict_resume_category(sample_jd_6)
        self.assertIn(result_6, [["Python Developer"], ["DevOps Engineer"]])

        # Test Case 7: Network Security Engineer JD
        sample_jd_7 = "Hiring a Network Security Engineer with expertise in firewalls, intrusion detection, and cybersecurity."
        result_7 = predict_resume_category(sample_jd_7)
        self.assertEqual(result_7, ["Network Security Engineer"])

        # Test Case 8: Automation Testing JD
        sample_jd_8 = "Looking for someone with Selenium, Jenkins, and test automation experience."
        result_8 = predict_resume_category(sample_jd_8)
        self.assertEqual(result_8, ["Automation Testing"])

        # Test Case 9: PMO JD
        sample_jd_9 = "Seeking a PMO with experience in project timelines, milestones, and office management."
        result_9 = predict_resume_category(sample_jd_9)
        self.assertEqual(result_9, ["PMO"])

        # Test Case 10: Electrical Engineering JD
        sample_jd_10 = "We are looking for an electrical engineer with experience in circuits, voltage systems, and transformers."
        result_10 = predict_resume_category(sample_jd_10)
        self.assertIn(result_10, [["Testing"], ["Electrical Engineering"]])
