const predictBtn = document.getElementById("predictBtn");

predictBtn.addEventListener("click", async function () {

    const airTemperature =
        parseFloat(document.getElementById("air_temperature").value);

    const processTemperature =
        parseFloat(document.getElementById("process_temperature").value);

    const rotationalSpeed =
        parseFloat(document.getElementById("rotational_speed").value);

    const torque =
        parseFloat(document.getElementById("torque").value);

    const toolWear =
        parseFloat(document.getElementById("tool_wear").value);


    // Validate input

    const values = [
        airTemperature,
        processTemperature,
        rotationalSpeed,
        torque,
        toolWear
    ];

    if (values.some(value => !Number.isFinite(value))) {
        alert("Please enter valid values for all machine sensors.");
        return;
    }


    try {

        predictBtn.textContent = "Analyzing...";
        predictBtn.disabled = true;


        const response = await fetch(
            "https://predictive-maintenance-ml-kem2.onrender.com/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    air_temperature: airTemperature,
                    process_temperature: processTemperature,
                    rotational_speed: rotationalSpeed,
                    torque: torque,
                    tool_wear: toolWear
                })
            }
        );


        if (!response.ok) {
            throw new Error("Prediction request failed");
        }


        const data = await response.json();


        // Display prediction

        const resultBox =
            document.getElementById("resultBox");

        const resultText =
            document.getElementById("resultText");

        const recommendation =
            document.getElementById("recommendation");

        const resultIcon =
            document.getElementById("resultIcon");

        const probability =
            document.getElementById("probability");

        const progressBar =
            document.getElementById("progressBar");


        probability.textContent =
            data.failure_probability + "%";

        progressBar.style.width =
            data.failure_probability + "%";

        resultText.textContent =
            data.result;

        recommendation.textContent =
            data.recommendation;


        if (data.prediction === 1) {

            resultBox.className = "result-box failure";

            resultIcon.textContent = "!";

        } else {

            resultBox.className = "result-box normal";

            resultIcon.textContent = "✓";
        }


        // Sensor summary

        document.getElementById("summaryAir").textContent =
            airTemperature + " K";

        document.getElementById("summaryProcess").textContent =
            processTemperature + " K";

        document.getElementById("summarySpeed").textContent =
            rotationalSpeed + " rpm";

        document.getElementById("summaryTorque").textContent =
            torque + " Nm";

        document.getElementById("summaryWear").textContent =
            toolWear + " min";


    } catch (error) {

        console.error(error);

        alert(
            "Could not connect to the prediction server. " +
            "Make sure FastAPI is running."
        );

    } finally {

        predictBtn.textContent = "Analyze Machine";
        predictBtn.disabled = false;

    }

});