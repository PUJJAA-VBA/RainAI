// Example: Animate precipitation percentage
document.addEventListener("DOMContentLoaded", () => {
  const circle = document.querySelector(".circle");
  let percent = 0;
  const target = 84;

  const interval = setInterval(() => {
    if (percent >= target) {
      clearInterval(interval);
    } else {
      percent++;
      circle.style.background =
        `conic-gradient(#2b8cff 0% ${percent}%, #e5e5e5 ${percent}% 100%)`;
      circle.querySelector("span").innerText = percent + "%";
    }
  }, 15);
});

async function getPrediction() {
    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            pressure: 1012,
            maxtemp: 34,
            temparature: 30,
            mintemp: 26,
            dewpoint: 24,
            humidity: 75,
            cloud: 40,
            sunshine: 8,
            windspeed: 18
        })
    });

    const data = await response.json();

    console.log(data.prediction);

    // Example: update some element in UI
    document.querySelector(".prediction-result").innerText = data.prediction;
}

function getPrediction() {
    const city = document.getElementById("cityInput").value;
    fetch(`http://127.0.0.1:5000/predict/${city}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = data.prediction;
        });
}

getPrediction();