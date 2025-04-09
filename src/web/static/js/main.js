// Additional JavaScript functionality for the Drug Interaction Assistant

// Function to format drug information
function formatDrugInfo(drugInfo) {
    if (!drugInfo) return '<p class="text-gray-500">No information available</p>';
    
    let html = `<h3 class="text-lg font-semibold mb-2">${drugInfo.drug_name}</h3>`;
    
    if (drugInfo.drug_class && drugInfo.drug_class !== "Information not available") {
        html += `<p class="mb-2"><span class="font-medium">Drug Class:</span> ${drugInfo.drug_class}</p>`;
    }
    
    if (drugInfo.mechanism && drugInfo.mechanism !== "Information not available") {
        html += `<p class="mb-2"><span class="font-medium">Mechanism:</span> ${drugInfo.mechanism}</p>`;
    }
    
    if (drugInfo.indications && drugInfo.indications.length > 0 && drugInfo.indications[0] !== "Information not available") {
        html += `<p class="font-medium mb-1">Indications:</p><ul class="list-disc pl-5 mb-2">`;
        drugInfo.indications.forEach(indication => {
            html += `<li>${indication}</li>`;
        });
        html += `</ul>`;
    }
    
    if (drugInfo.side_effects && drugInfo.side_effects.length > 0 && drugInfo.side_effects[0] !== "Information not available") {
        html += `<p class="font-medium mb-1">Side Effects:</p><ul class="list-disc pl-5 mb-2">`;
        drugInfo.side_effects.forEach(effect => {
            html += `<li>${effect}</li>`;
        });
        html += `</ul>`;
    }
    
    if (drugInfo.common_interactions && drugInfo.common_interactions.length > 0 && drugInfo.common_interactions[0] !== "Information not available") {
        html += `<p class="font-medium mb-1">Common Interactions:</p><ul class="list-disc pl-5 mb-2">`;
        drugInfo.common_interactions.forEach(interaction => {
            html += `<li>${interaction}</li>`;
        });
        html += `</ul>`;
    }
    
    if (drugInfo.contraindications && drugInfo.contraindications.length > 0 && drugInfo.contraindications[0] !== "Information not available") {
        html += `<p class="font-medium mb-1">Contraindications:</p><ul class="list-disc pl-5 mb-2">`;
        drugInfo.contraindications.forEach(contraindication => {
            html += `<li>${contraindication}</li>`;
        });
        html += `</ul>`;
    }
    
    return html;
}

// Function to extract drug names from a message
function extractDrugNames(message) {
    // Simple regex to find drug names in common question patterns
    const patterns = [
        /(?:between|with|and)\s+([A-Za-z-]+)\s+(?:and|with)\s+([A-Za-z-]+)/i,
        /(?:about|information on|details about)\s+([A-Za-z-]+)/i,
        /(?:visualization|graph|network)\s+(?:for|of)\s+([A-Za-z-]+)/i
    ];
    
    for (const pattern of patterns) {
        const match = message.match(pattern);
        if (match) {
            // If we have two drug names, return both
            if (match.length > 2) {
                return [match[1], match[2]];
            }
            // Otherwise return just the one drug name
            return [match[1]];
        }
    }
    
    return [];
}

// Function to fetch drug information
function fetchDrugInfo(drugName) {
    const drugInfoContainer = document.getElementById('drug-info-container');
    
    // Show loading state
    drugInfoContainer.innerHTML = '<div class="loading"><div></div><div></div><div></div><div></div></div>';
    
    // Fetch drug information
    fetch(`/api/drug-info/${drugName}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                drugInfoContainer.innerHTML = `<p class="text-red-500">${data.error}</p>`;
            } else {
                drugInfoContainer.innerHTML = formatDrugInfo(data);
            }
        })
        .catch(err => {
            drugInfoContainer.innerHTML = '<p class="text-red-500">Error fetching drug information</p>';
            console.error(err);
        });
}

// Function to fetch visualization
function fetchVisualization(drugName) {
    const vizContainer = document.getElementById('visualization-container');
    
    // Show loading state
    vizContainer.innerHTML = '<div class="loading"><div></div><div></div><div></div><div></div></div>';
    
    // Fetch visualization
    fetch(`/api/visualize/${drugName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Visualization not available');
            }
            return response.blob();
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            vizContainer.innerHTML = `<img src="${url}" alt="Drug interaction visualization" class="max-w-full max-h-full">`;
        })
        .catch(err => {
            vizContainer.innerHTML = '<p class="text-red-500">Error generating visualization</p>';
            console.error(err);
        });
}

// Enhance the existing message processing
document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    
    // Add event listener for input changes to detect drug names
    userInput.addEventListener('input', () => {
        const drugNames = extractDrugNames(userInput.value);
        
        if (drugNames.length === 1) {
            // If we detect a single drug name, fetch its information
            fetchDrugInfo(drugNames[0]);
        }
    });
    
    // Enhance the existing send button click handler
    const sendBtn = document.getElementById('send-btn');
    const originalClickHandler = sendBtn.onclick;
    
    sendBtn.onclick = (e) => {
        // Call the original handler if it exists
        if (originalClickHandler) {
            originalClickHandler(e);
        }
        
        // Extract drug names from the message
        const drugNames = extractDrugNames(userInput.value);
        
        if (drugNames.length === 1) {
            // If we detect a single drug name, fetch its visualization
            fetchVisualization(drugNames[0]);
        }
    };
}); 