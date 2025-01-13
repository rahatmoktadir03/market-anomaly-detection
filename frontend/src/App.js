import React, { useState } from "react";
import axios from "axios";

const App = () => {
  const [inputData, setInputData] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle form input
  const handleInputChange = (e) => {
    setInputData(e.target.value);
  };

  // Call the Flask backend API to get the prediction
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Send a POST request to the backend API
      const response = await axios.post("http://localhost:5000/predict", {
        data: inputData,
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Market Anomaly Detection</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Enter Financial Data:
          <input
            type="text"
            value={inputData}
            onChange={handleInputChange}
            placeholder="Enter data for prediction"
          />
        </label>
        <button type="submit" disabled={loading}>
          Get Prediction
        </button>
      </form>

      {loading && <p>Loading...</p>}

      {prediction && (
        <div>
          <h2>Prediction:</h2>
          <p>
            {prediction === 1 ? "Market Crash Detected!" : "Market is Stable."}
          </p>
        </div>
      )}
    </div>
  );
};

export default App;
