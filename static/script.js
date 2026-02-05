async function analyzeMovie() {
  let url = document.getElementById("url").value;
  document.getElementById("result").innerHTML = `<p>⏳ Fetching reviews...</p>`;
  
  let response = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: url })
  });

  let data = await response.json();

  if (data.error) {
    document.getElementById("result").innerHTML = `<p class="error">${data.error}</p>`;
    return;
  }

  document.getElementById("result").innerHTML = `
    <h2> Overall Result</h2>
    <p><b>Movie Used:</b> ${data.movie_used}</p>
    <p><b>Sentiment:</b> ${data.overall_sentiment}</p>
    <p><b>Positive:</b> ${data.positive_percent}%</p>
    <p><b>Negative:</b> ${data.negative_percent}%</p>
    <p><b>Total Reviews:</b> ${data.total_reviews}</p>
    <p><b>Keywords:</b> ${data.keywords.join(", ")}</p>
  `;
}
async function uploadCSV(){
  let fileInput = document.getElementById("csvFile");
  let file = fileInput.files[0];

  if(!file){
    alert("Select a CSV file first!");
    return;
  }
  else{
    alert("Uploaded");
  }

  let formData = new FormData();
  formData.append("file", file);

  let res = await fetch("/upload_csv", {
    method: "POST",
    body: formData
  });

  let data = await res.json();
  alert(data.message || data.error);
}
