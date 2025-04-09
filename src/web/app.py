"""Flask web application for the drug interaction system."""

import os
import uuid
import io
from flask import Flask, request, jsonify, render_template, send_file
import matplotlib.pyplot as plt

from ..models.chatbot import DrugInteractionChatbot

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Initialize the chatbot
    chatbot = DrugInteractionChatbot()
    
    # Ensure the visualization directory exists
    os.makedirs("static/visualizations", exist_ok=True)
    
    @app.route('/')
    def home():
        """Render the home page"""
        return render_template('index.html')
    
    @app.route('/api/ask', methods=['POST'])
    def ask():
        """Process a user message and return a response"""
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chatbot.process_message(user_message)
        
        # Check if we need to generate a visualization
        visualization_needed = False
        drug_name = None
        
        if "interaction found between" in response:
            # Extract drug name from response
            import re
            match = re.search(r'interaction found between (.+?) and', response)
            if match:
                drug_name = match.group(1)
                visualization_needed = True
        
        result = {
            'response': response,
            'visualization': None
        }
        
        if visualization_needed and drug_name:
            # Generate a unique ID for this visualization
            viz_id = str(uuid.uuid4())
            
            # Create and save the visualization
            G, error = chatbot.processor.generate_network(drug_name)
            
            if not error:
                # Save the visualization to a file
                plt.savefig(f"static/visualizations/{viz_id}.png")
                plt.close()
                
                # Add the URL to the result
                result['visualization'] = f"/static/visualizations/{viz_id}.png"
        
        return jsonify(result)
    
    @app.route('/api/visualize/<drug_name>')
    def visualize(drug_name):
        """Generate a visualization for a specific drug"""
        # Create the visualization
        G, error = chatbot.processor.generate_network(drug_name)
        
        if error:
            return jsonify({'error': error}), 404
        
        # Save to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return send_file(buf, mimetype='image/png')
    
    @app.route('/api/analyze-note', methods=['POST'])
    def analyze_note():
        """Analyze a clinical note for drug interactions"""
        data = request.json
        clinical_note = data.get('note', '')
        
        if not clinical_note:
            return jsonify({'error': 'No clinical note provided'}), 400
        
        # Extract medications and interactions from note
        results = chatbot.processor.extract_drugs_from_clinical_notes(clinical_note)
        
        # Enhance results with database information
        meds = [med.get("name") for med in results["medications"] if med.get("name")]
        db_interactions = []
        
        # Check all pairs of medications
        for i in range(len(meds)):
            for j in range(i+1, len(meds)):
                interactions, _ = chatbot.db.get_interactions(meds[i], meds[j])
                for d1, d2, desc, severity, source in interactions:
                    db_interactions.append({
                        "drug1": meds[i],
                        "drug2": meds[j],
                        "description": desc,
                        "severity": severity,
                        "source": source
                    })
        
        # Add database interactions to results
        results["database_interactions"] = db_interactions
        
        return jsonify(results)
    
    @app.route('/api/drug-info/<drug_name>')
    def drug_info(drug_name):
        """Get information about a specific drug"""
        # Get drug information
        drug_info = chatbot.processor.get_drug_information(drug_name)
        
        if not drug_info:
            return jsonify({'error': f'No information found for {drug_name}'}), 404
            
        return jsonify(drug_info)
    
    @app.route('/api/interaction-network')
    def interaction_network():
        """Generate a network visualization of drug interactions"""
        drug_name = request.args.get('drug', None)
        depth = int(request.args.get('depth', 1))
        
        if drug_name:
            G, error = chatbot.processor.generate_network(drug_name, depth)
        else:
            G, error = chatbot.processor.generate_network()
            
        if error:
            return jsonify({'error': error}), 404
            
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Get positions
        import networkx as nx
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_sizes = [G.nodes[node].get('size', 10) for node in G.nodes()]
        node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Draw edges with colors based on severity
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            edge_colors.append(data.get('color', 'gray'))
            edge_widths.append(data.get('weight', 1))
            
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        # Save to BytesIO
        buf = io.BytesIO()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return send_file(buf, mimetype='image/png')
    
    return app

def run_app():
    """Run the Flask application"""
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    run_app() 