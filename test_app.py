import unittest
import json
from app import app

class TestPredictEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        payload = {
            'features': [
                0.00632, 18.0, 2.31, 0.0,
                0.538, 6.575, 65.2, 4.0900,
                1.0, 296.0, 15.3, 396.90, 4.98
            ]
        }

        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn('predictions', data)
        
if __name__ == '__main__':
    unittest.main()
