function getPrediction() {

    const city = document.getElementById("cityInput").value;

    if (!city) {
        alert("Please enter city name");
        return;
    }

    fetch(`http://127.0.0.1:5000/predict/${city}`)
    .then(response => response.json())
    .then(data => {

        console.log(data);

        document.getElementById("result").innerText =
            data.prediction || "Prediction failed";

    })
    .catch(error => {
        console.log(error);
        document.getElementById("result").innerText =
            "Server error";
    });

}