"""
Test Flask App
"""
import unittest
import json
from app import app

class TestPredictEndpoint(unittest.TestCase):
    """
    A test case for the predict endpoint of the Flask application.
    """
    def setUp(self):
        """
        Set up the test client and enable testing mode.
        This method is called before each test to prepare the environment.
        """
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        """
        Test the predict endpoint with a sample payload.
        
        This test sends a POST request to the /predict endpoint with a JSON payload
        containing features for prediction. It checks if the response status code 
        is 200, if the response is in JSON format, and if the 'predictions' key is 
        present in the response data.
        """
        payload = {
            'features': [
                0.00632, 18.0, 2.31, 0.0,
                0.538, 6.575, 65.2, 4.0900,
                1.0, 296.0, 15.3, 396.90, 4.98
            ]
        }
        headers = {'Content-Type':'application/json'}
        response = self.app.post('/predict', data=json.dumps(payload), headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn('predictions', data)

if __name__ == '__main__':
    unittest.main()
