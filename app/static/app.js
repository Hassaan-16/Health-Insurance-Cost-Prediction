const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const meta = document.getElementById("meta");
const retrainBtn = document.getElementById("retrain-btn");

function formatCurrency(value) {
  return `$${Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function setResult(text, type = "") {
  result.className = `result ${type}`.trim();
  result.textContent = text;
}

async function loadModelInfo() {
  const res = await fetch("/api/model-info");
  const data = await res.json();
  meta.textContent = JSON.stringify(data, null, 2);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  payload.age = Number(payload.age);
  payload.bmi = Number(payload.bmi);
  payload.children = Number(payload.children);
  if (payload.actual_charge === "") {
    delete payload.actual_charge;
  } else if (payload.actual_charge !== undefined) {
    payload.actual_charge = Number(payload.actual_charge);
  }

  setResult("Calculating estimate...");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      setResult(data.error || "Prediction failed.", "error");
      return;
    }

    const lines = [
      `Predicted annual charge: ${formatCurrency(data.predicted_charge)} (${data.risk_band} risk band)`,
    ];

    if (data.actual_charge !== null && data.actual_charge !== undefined) {
      lines.push(
        `Actual annual charge (${data.actual_charge_source}): ${formatCurrency(data.actual_charge)}`
      );
      lines.push(`Difference (predicted - actual): ${formatCurrency(data.difference)}`);
      lines.push(`Absolute difference: ${formatCurrency(data.absolute_difference)}`);
    } else {
      lines.push("Actual annual charge: Not available (no matching dataset row and no manual value).");
    }

    setResult(lines.join("\n"));
  } catch (err) {
    setResult("Server request failed.", "error");
  }
});

retrainBtn.addEventListener("click", async () => {
  retrainBtn.disabled = true;
  retrainBtn.textContent = "Retraining...";

  try {
    const res = await fetch("/api/retrain", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      setResult(data.error || "Retraining failed.", "error");
    } else {
      setResult("Model retrained successfully.");
      await loadModelInfo();
    }
  } catch (err) {
    setResult("Retraining request failed.", "error");
  } finally {
    retrainBtn.disabled = false;
    retrainBtn.textContent = "Retrain Model";
  }
});

loadModelInfo().catch(() => {
  meta.textContent = "Could not load model metadata.";
});
