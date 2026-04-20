let doseChartInstance = null;

// Animated Counter Utility
function animateValue(obj, start, end, duration, decimals = 2) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        let currentVal = progress * (end - start) + start;
        obj.innerHTML = currentVal.toFixed(decimals);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerHTML = end.toFixed(decimals);
        }
    };
    window.requestAnimationFrame(step);
}

document.getElementById('dose-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI state loading & Hologram Scanning FX
    const btn = document.getElementById('calculate-btn');
    const btnText = btn.querySelector('.btn-text');

    btn.classList.add('processing');
    btnText.innerText = "PROCESSING ALGORITHM...";

    // Trigger visual scanner
    const scanLines = document.getElementById('scan-fx');
    scanLines.classList.remove('scanning');
    void scanLines.offsetWidth; // trigger reflow
    scanLines.classList.add('scanning');

    // Gather payload
    const payload = {
        species: document.getElementById('species').value,
        weight: parseFloat(document.getElementById('weight').value),
        age: parseFloat(document.getElementById('age').value),
        rbc: parseFloat(document.getElementById('rbc').value),
        hb: parseFloat(document.getElementById('hb').value),
        creatinine: parseFloat(document.getElementById('creatinine').value),
        glucose: parseFloat(document.getElementById('glucose').value),
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Network error during prediction.");
        const data = await response.json();

        // Wait a slight bit for the scan line animation to feel impactful
        setTimeout(() => {
            updateDashboard(data);

            btn.classList.remove('processing');
            btnText.innerText = "INITIALIZE ML SEQUENCE";
            scanLines.classList.remove('scanning');
        }, 800);

    } catch (error) {
        console.error("Error calling ML API:", error);
        alert("CRITICAL ERROR: Unable to reach the Neural Network Engine.");
        btn.classList.remove('processing');
        btnText.innerText = "INITIALIZE ML SEQUENCE";
        scanLines.classList.remove('scanning');
    }
});

function updateDashboard(data) {
    const displayPanel = document.getElementById('results-display');

    // Enable Active State
    if (displayPanel.classList.contains('idle')) {
        displayPanel.classList.remove('idle');
        displayPanel.classList.add('active');
    }

    // Dynamic Number Counters
    const optimalEl = document.getElementById('optimal-dose');
    const maxProbEl = document.getElementById('max-prob');

    animateValue(optimalEl, 0, data.optimal_dose, 1000, 2);
    document.getElementById('safe-lower').innerText = data.safe_lower.toFixed(2);
    document.getElementById('safe-upper').innerText = data.safe_upper.toFixed(2);
    animateValue(maxProbEl, 0, data.max_prob, 1000, 1);

    // Status Banner Logic
    const statusFlag = document.getElementById('status-flag');
    statusFlag.innerText = "[" + data.status.toUpperCase() + "]";
    if (data.status.toLowerCase().includes('safe')) {
        statusFlag.className = 'badge status-safe';
    } else {
        statusFlag.className = 'badge status-risk';
    }

    // Render Cybernetic Chart
    renderChart(data.chart_labels, data.chart_data);
}

function renderChart(labels, dataPoints) {
    const ctx = document.getElementById('doseChart').getContext('2d');

    if (doseChartInstance) {
        doseChartInstance.destroy();
    }

    // Futuristic Neon Gradient
    const gradientFill = ctx.createLinearGradient(0, 0, 0, 300);
    gradientFill.addColorStop(0, 'rgba(0, 240, 255, 0.6)');
    gradientFill.addColorStop(1, 'rgba(0, 240, 255, 0.01)');

    doseChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Safety Prob (%)',
                data: dataPoints,
                borderColor: '#00F0FF',
                backgroundColor: gradientFill,
                borderWidth: 2,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#00f0ff',
                pointRadius: 0,
                pointHitRadius: 15,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(5, 8, 20, 0.9)',
                    titleColor: '#00f0ff',
                    titleFont: { family: 'Orbitron', size: 12 },
                    bodyColor: '#fff',
                    bodyFont: { family: 'Rajdhani', size: 14 },
                    borderColor: '#00f0ff',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function (context) {
                            return 'PROBABILITY: ' + context.parsed.y.toFixed(1) + '%';
                        },
                        title: function (context) {
                            return 'DOSE VECTOR: ' + context[0].label + ' mg/kg';
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'ADMINISTERED XYLAZINE (mg/kg)', color: '#64748b', font: { family: 'Orbitron', size: 10 } },
                    grid: { color: 'rgba(0, 240, 255, 0.08)', drawBorder: false },
                    ticks: { color: '#64748b', font: { family: 'Rajdhani', size: 12 } }
                },
                y: {
                    grid: { color: 'rgba(0, 240, 255, 0.08)', drawBorder: false },
                    ticks: { color: '#64748b', font: { family: 'Rajdhani', size: 12 } },
                    min: 0,
                    max: 100
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}
