// Global variable to hold the chart instance so we can destroy/recreate it on new uploads
let chartInstance = null;

// Listen for the "Initiate Scan" / "Analyze Traffic" button click
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('csvFileInput');
    const loadingDiv = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const uploadBtn = document.getElementById('uploadBtn');

    // 1. Validation
    if (fileInput.files.length === 0) {
        alert("⚠️ Please select a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // 2. UI State: Show loading, hide previous results, disable button
    loadingDiv.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    uploadBtn.disabled = true;
    
    // Update button text if the span exists (from our new HTML structure)
    const btnText = uploadBtn.querySelector('.btn-text');
    if (btnText) {
        btnText.innerText = "Scanning...";
    } else {
        uploadBtn.innerText = "Scanning..."; // Fallback
    }

    // 3. Make the API Request
    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${errorText}`);
        }

        const data = await response.json();
        
        // 4. Render the results
        renderChart(data.summary);
        renderTable(data.results);
        
        // 5. UI State: Show results and scroll to them
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error("Error:", error);
        alert("❌ Failed to process traffic: " + error.message);
    } finally {
        // 6. UI State: Cleanup loading state
        loadingDiv.classList.add('hidden');
        uploadBtn.disabled = false;
        if (btnText) {
            btnText.innerText = "Initiate Scan";
        } else {
            uploadBtn.innerText = "Initiate Scan";
        }
    }
});

function renderChart(summaryData) {
    const ctx = document.getElementById('summaryChart').getContext('2d');
    
    // Destroy previous chart if it exists to prevent overlap issues
    if (chartInstance) {
        chartInstance.destroy();
    }

    const labels = Object.keys(summaryData);
    const data = Object.values(summaryData);

    // Neon Red for attack, Neon Green for benign
    const backgroundColors = labels.map(label => 
        label.includes("ATTACK") ? '#ef4444' : '#10b981'
    );

    chartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderWidth: 2,
                borderColor: '#1e293b', // Matches the dark inner card background
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { 
                    position: 'bottom',
                    labels: { 
                        color: '#f8fafc', // Light text for dark background readability
                        font: { family: "'Poppins', sans-serif" } 
                    }
                }
            }
        }
    });
}

function renderTable(results) {
    const tableHeaders = document.getElementById('tableHeaders');
    const tableBody = document.getElementById('tableBody');
    
    // Clear previous table data
    tableHeaders.innerHTML = '';
    tableBody.innerHTML = '';

    if (results.length === 0) return;

    // --- Build Headers ---
    const headers = Object.keys(results[0]);
    headers.forEach(headerText => {
        const th = document.createElement('th');
        // Replace underscores with spaces for a cleaner look
        th.textContent = headerText.replace(/_/g, ' '); 
        tableHeaders.appendChild(th);
    });

    // --- Build Rows ---
    // Limit to 100 rows so the browser doesn't freeze on massive files
    const displayLimit = Math.min(results.length, 100);
    
    for (let i = 0; i < displayLimit; i++) {
        const row = results[i];
        const tr = document.createElement('tr');
        
        headers.forEach(header => {
            const td = document.createElement('td');
            
            // If this is the prediction column, style it as a badge
            if (header === 'DCNN_Prediction' || header === 'DCNN Prediction') {
                const span = document.createElement('span');
                span.textContent = row[header];
                
                // Check if it's an attack or benign and assign the right CSS class
                if (row[header].includes("ATTACK")) {
                    span.className = "badge badge-attack";
                } else {
                    span.className = "badge badge-benign";
                }
                
                td.appendChild(span);
            } else {
                // Regular text for other columns
                td.textContent = row[header];
            }
            tr.appendChild(td);
        });

        tableBody.appendChild(tr);
    }
}