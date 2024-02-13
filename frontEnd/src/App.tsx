// Importing necessary modules and components
import axios from "axios"; // Library for making HTTP requests
import { useRef, useState } from "react"; // React hooks for managing state and references
import CanvasDraw from "react-canvas-draw"; // Canvas drawing component for React

// Main App component
function App() {
  // Reference to the CanvasDraw component
  const ref = useRef<CanvasDraw>(null);

  // State to store the predicted number
  const [predictedNumber, setPredictedNumber] = useState<number[]>([0, 0]);
  const [trainingData, setTrainingData] = useState<number[][]>([]);
  const [trainingNumber, setTrainingNumber] = useState<number>(0);


  // Function to request prediction based on the drawn image
  const requestPrediction = async(base64: string)=>{
    // Replace with the actual prediction URL
    const URL_HERE = "http://127.0.0.1:5000/predict";

    // Sending the image data to the prediction API
    const res = await axios.post(URL_HERE, {
      image:base64.split(",")[1] // Extracting the image data (excluding metadata)
    });

    // checking bug
    // console.log(res.data.predictions);

    // Checking if the predictions array has elements
    if(res.data.predictions?.length){
      // Logging the first prediction to the console
      // console.log(res.data.predictions[0]);

      // Setting the predicted number in the state
      setPredictedNumber(res.data.predictions);
    }
  }


  // Function to send training data
  const sendTraining = async (trainNumber: number[], imageURL: string) => {
    const URL_HERE = "http://127.0.0.1:5000/train";

    // Processing the imageURL
    const base64Image = imageURL.split(",")[1]; // Extracting the image data (excluding metadata)
    // console.log(base64Image)

    const res = await axios.post(URL_HERE, {
      image: base64Image,
      trainNumber: trainNumber[0]
    })

    if (res.status === 200) {
      console.log("Training data sent successfully!");
    } else {
      console.error("Training data not sent. Server responded with error:", res.data.error);
    }
  }


  // Function to clear the drawing on the canvas
  const onClear = ()=>{
    if(ref.current){
      ref.current.clear();
    }

    // reset the prediction number storage
    setPredictedNumber([0, 0]);
  };


  // Function to handle the prediction request when the "Predict" button is clicked
  const onClickSend = ()=>{
    if(ref.current){
      // Getting the data URL of the drawn image
      const imageURL = ref.current.getDataURL("png", false, "0xffffff");

      // Temporary log to display the image data URL
      // console.log(imageURL);

      // Sending the drawn image for prediction
      requestPrediction(imageURL);
    }
  };


  // New feature
  const addTrainingData = () => {
    if (ref.current) {
      const imageURL = ref.current.getDataURL("png", false, "0xffffff");
      const newTrainingData = [...trainingData, [trainingNumber, ...predictedNumber]];
      setTrainingData(newTrainingData);

      const trainNumber = newTrainingData[newTrainingData.length - 1]
      sendTraining(trainNumber, imageURL)
    }
  };

  return <div style={{display:"flex", flexDirection:"column", gap:9, padding:5}}>
    <h2 style={{lineHeight: 0}}>Draw Your Number</h2>

    {/* Canvas */}
    <CanvasDraw
      canvasHeight={300}
      canvasWidth={400}
      ref={ref}
    />

    <div>
      <button
        onClick={()=>{
          onClear()
        }}
      >
        Clear
      </button>

      <button
        onClick={()=>{
          onClickSend();
        }}
      >
        Predict
      </button>
    </div>

    <div>
        <p>Prediction number is {predictedNumber[0]} or maybe {predictedNumber[1]}</p>
        <input type="number" placeholder="Enter actual number" value={trainingNumber} onChange={(e) => setTrainingNumber(parseInt(e.target.value))} />
        <button onClick={addTrainingData}>Add Training Data</button>
    </div>

    <div>
      <h3>Training Data</h3>
      <ul>
        {trainingData.map((data, index) => (
          <li key={index}>Actual: {data[0]}, Predicted: {data[1]}</li>
        ))}
      </ul>
    </div>

  </div>
}
	
export default App;


