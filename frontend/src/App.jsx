import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"

function App() {
  // --- State variables ---

  // Stores the .pkl model file object selected by the user
  const [modelFile, setModelFile] = useState(null)
  // Stores the .csv dataset file object selected by the user
  const [datasetFile, setDatasetFile] = useState(null)
  // Stores the JSON response from FastAPI after analysis completes
  const [result, setResult] = useState(null)
  // True while waiting for FastAPI response — disables button to prevent double clicks
  const [loading, setLoading] = useState(false)
  // Stores any error message to display to the user
  const [error, setError] = useState(null)

  /**
   * handleSubmit — called when user clicks "Run Analysis"
   * Packages the two files into a FormData object and sends them
   * to the FastAPI /explain/ endpoint via a POST request
   */
  const handleSubmit = async () => {
    // Validate both files are selected before sending
    if (!modelFile || !datasetFile) {
      setError("Please upload both a model file and a dataset file.")
      return
    }

    setLoading(true)  // Show "Analyzing..." on button
    setError(null)    // Clear any previous error message

    // FormData is the standard way to send files over HTTP
    const formData = new FormData()
    formData.append("model_file", modelFile)     // must match FastAPI parameter name
    formData.append("dataset_file", datasetFile) // must match FastAPI parameter name

    try {
      // Send POST request to FastAPI /explain/ endpoint
      const response = await fetch("http://localhost:8000/explain/", {
        method: "POST",
        body: formData,  // files are sent in the request body
      })
      const data = await response.json()  // parse the JSON response
      setResult(data)  // store result to trigger UI update
    } catch (err) {
      // Catches network errors e.g. FastAPI not running
      setError("Something went wrong. Make sure FastAPI is running.")
    } finally {
      // Always runs after try/catch regardless of success or failure
      setLoading(false)  // Re-enable the button
    }
  }

  /**
   * getChartData — transforms raw SHAP values into chart-friendly format
   * 
   * SHAP returns one value per feature per sample, which is too granular to plot directly.
   * This function computes the mean absolute SHAP value per feature:
   * - Absolute value: captures importance regardless of direction (positive/negative)
   * - Mean: averages across all samples to get one importance score per feature
   * 
   * Returns an array like: [{ feature: "petal length (cm)", importance: 0.32 }, ...]
   * sorted from most to least important
   */

  const getChartData = (result) => {
    if (!result) return []

    const shapValues = result.shap_values
    const featureNames = result.feature_names

    const meanAbsShap = featureNames.map((name, i) => {
      let values = []

      // Check if 3D array: (samples, features, classes)
      if (Array.isArray(shapValues[0][0])) {
        // 3D: shapValues[sample][feature][class]
        // we want all values for feature i across all samples and classes
        shapValues.forEach(sample => {
          sample[i].forEach(classVal => {
            values.push(Math.abs(classVal))
          })
        })
      } else {
        // 2D: shapValues[sample][feature]
        shapValues.forEach(sample => {
          values.push(Math.abs(sample[i]))
        })
      }

      const mean = values.reduce((a, b) => a + b, 0) / values.length
      return { feature: name, importance: parseFloat(mean.toFixed(4)) }
    })

    return meanAbsShap.sort((a, b) => b.importance - a.importance)
  }

  // Pre-compute chart data whenever result changes
  const chartData = getChartData(result)

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center py-12 px-4">
      <h1 className="text-4xl font-bold mb-2">See Inside the Black Box</h1>
      <p className="text-gray-400 mb-10">Upload your ML model and dataset to get SHAP explanations</p>

      <div className="bg-gray-800 rounded-xl p-8 w-full max-w-lg shadow-lg">
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Upload Model (.pkl)</label>
          <input
            type="file"
            accept=".pkl"
            onChange={(e) => setModelFile(e.target.files[0])}
            className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700 cursor-pointer"
          />
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Upload Dataset (.csv)</label>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setDatasetFile(e.target.files[0])}
            className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700 cursor-pointer"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white font-semibold py-3 rounded-lg transition-colors"
        >
          {loading ? "Analyzing..." : "Run Analysis"}
        </button>

        {error && <p className="mt-4 text-red-400 text-sm">{error}</p>}
      </div>

      {result && (
        <div className="mt-10 w-full max-w-2xl bg-gray-800 rounded-xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold mb-4">Results</h2>
          <div className="flex gap-6 mb-6 text-sm text-gray-300">
            <p>Model: <span className="text-white font-medium">{result.model_type}</span></p>
            <p>Samples: <span className="text-white font-medium">{result.samples_processed}</span></p>
            <p>Time: <span className="text-white font-medium">{result.processing_time_seconds}s</span></p>
          </div>

          <h3 className="text-lg font-semibold mb-4">Feature Importance (Mean |SHAP Value|)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} layout="vertical">
              <XAxis type="number" stroke="#9ca3af" />
              <YAxis type="category" dataKey="feature" width={150} stroke="#9ca3af" />
              <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "none", borderRadius: "8px" }} />
              <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

export default App