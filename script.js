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

function getPrediction() {

    const city = document.getElementById("cityInput").value;

    fetch(`http://127.0.0.1:5000/predict/${city}`)
    .then(response => response.json())
    .then(data => {

        console.log(data);

        if(data.prediction){
            document.getElementById("result").innerText =
                data.prediction;
        }
        else{
            document.getElementById("result").innerText =
                "Prediction failed";
        }

    })
    .catch(err => {
        console.log(err);
        document.getElementById("result").innerText =
            "Server error";
    });

}