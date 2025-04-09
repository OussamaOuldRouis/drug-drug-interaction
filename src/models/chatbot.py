"""Drug Interaction Chatbot for natural language interaction with the drug interaction system."""

import re
import uuid
import io
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any

from .biomedical_llm import BiomedicalLLM
from .drug_interaction_db import DrugInteractionDatabase
from .ddi_processor import DDIProcessor

class DrugInteractionChatbot:
    """Chatbot interface for drug interaction analysis system."""
    
    def __init__(self, model_name="stanford-crfm/BioMedLM"):
        """Initialize the Drug Interaction Chatbot with Biomedical LLM"""
        self.db = DrugInteractionDatabase()
        self.bio_llm = BiomedicalLLM(model_name)
        self.processor = DDIProcessor(self.db, self.bio_llm)
        
    def process_message(self, message):
        """Process a user message and provide an appropriate response"""
        # Check if this is a clinical notes analysis request
        if any(term in message.lower() for term in ["clinical note", "patient note", "extract from", "analyze note", "medical record"]):
            # Extract the clinical note part
            note_pattern = r"(?:clinical note|patient note|medical record)[s]?:?\s*([\s\S]+)$"
            note_match = re.search(note_pattern, message, re.IGNORECASE)
            
            if note_match:
                clinical_text = note_match.group(1).strip()
                extracted_info = self.processor.extract_drugs_from_clinical_notes(clinical_text)
                
                # Format the response
                response = "📋 **Analysis of Clinical Notes**\n\n"
                
                # Add medications
                if extracted_info["medications"]:
                    response += "**Medications Identified:**\n"
                    for med in extracted_info["medications"]:
                        name = med.get("name", "Unknown")
                        dosage = med.get("dosage", "Not specified")
                        frequency = med.get("frequency", "Not specified")
                        
                        if dosage != "Not specified" or frequency != "Not specified":
                            response += f"- {name}: {dosage} {frequency}\n"
                        else:
                            response += f"- {name}\n"
                    response += "\n"
                else:
                    response += "No medications were identified in the clinical notes.\n\n"
                
                # Add potential interactions
                if extracted_info["potential_interactions"]:
                    response += "**Potential Interactions:**\n"
                    for interaction in extracted_info["potential_interactions"]:
                        drug1 = interaction.get("drug1", "Unknown")
                        drug2 = interaction.get("drug2", "Unknown")
                        concern = interaction.get("concern", "Potential interaction")
                        
                        response += f"- {drug1} + {drug2}: {concern}\n"
                    response += "\n"
                else:
                    # Try to identify interactions from the medications list
                    meds = [med.get("name") for med in extracted_info["medications"] if med.get("name")]
                    potential_interactions = []
                    
                    # Check all pairs of medications
                    for i in range(len(meds)):
                        for j in range(i+1, len(meds)):
                            interactions, _ = self.db.get_interactions(meds[i], meds[j])
                            if interactions:
                                for d1, d2, desc, severity, _ in interactions:
                                    potential_interactions.append(f"- {meds[i]} + {meds[j]}: {desc} ({severity})")
                    
                    if potential_interactions:
                        response += "**Potential Interactions:**\n"
                        response += "\n".join(potential_interactions) + "\n\n"
                    else:
                        response += "No potential interactions were identified among the medications.\n\n"
                
                response += "Please consult with a healthcare professional for a comprehensive review of drug interactions and medical advice."
                
                return response
        
        # Check if user is asking for information about a specific drug
        drug_info_pattern = r"(tell me about|information on|what is|info about|details on)\s+(.+?)(?:\?|$)"
        drug_info_match = re.search(drug_info_pattern, message.lower())
        
        if drug_info_match:
            drug_name = drug_info_match.group(2).strip()
            canonical = self.db.search_drug(drug_name)
            
            # If not in database, use original name but still try to get info
            if not canonical:
                canonical = drug_name
            
            # Get drug information from biomedical LLM
            drug_info = self.processor.get_drug_information(canonical)
            
            if drug_info:
                # Format the response
                response = f"📊 **Information about {drug_info['drug_name']}**\n\n"
                
                if drug_info.get("drug_class") and drug_info["drug_class"] != "Information not available":
                    response += f"**Drug Class:** {drug_info['drug_class']}\n\n"
                
                if drug_info.get("mechanism") and drug_info["mechanism"] != "Information not available":
                    response += f"**Mechanism of Action:** {drug_info['mechanism']}\n\n"
                
                if drug_info.get("indications") and drug_info["indications"][0] != "Information not available":
                    response += "**Common Indications:**\n"
                    for indication in drug_info["indications"]:
                        response += f"- {indication}\n"
                    response += "\n"
                
                if drug_info.get("side_effects") and drug_info["side_effects"][0] != "Information not available":
                    response += "**Common Side Effects:**\n"
                    for effect in drug_info["side_effects"]:
                        response += f"- {effect}\n"
                    response += "\n"
                
                if drug_info.get("common_interactions") and drug_info["common_interactions"][0] != "Information not available":
                    response += "**Common Interactions:**\n"
                    for interaction in drug_info["common_interactions"]:
                        response += f"- {interaction}\n"
                    response += "\n"
                
                if drug_info.get("contraindications") and drug_info["contraindications"][0] != "Information not available":
                    response += "**Contraindications:**\n"
                    for contraindication in drug_info["contraindications"]:
                        response += f"- {contraindication}\n"
                    response += "\n"
                
                response += "This information is for educational purposes only. Always consult a healthcare professional for medical advice."
                
                return response
                
            else:
                return f"I couldn't find detailed information about {drug_name}. Please check the spelling or try another medication."
        
        # Check if this is a drug interaction query
        if re.search(r'take|interact|safe|drug|interaction|medicine|pill|medication', message.lower()):
            result = self.processor.process_query(message)
            
            if result["status"] == "error":
                return result["message"]
            
            elif result["status"] == "not_found":
                return result["message"]
            
            elif result["status"] == "no_interaction":
                return (f"Based on our database and biomedical literature analysis, no known interactions were found between {result['drugs'][0]} "
                       f"and {result['drugs'][1]}. However, always consult with a healthcare "
                       f"professional before combining medications.")
            
            elif result["status"] == "found":
                drug1, drug2 = result['drugs']
                interactions = result["interactions"]
                
                # Generate response
                response = f"⚠️ **Potential interaction found between {drug1} and {drug2}:**\n\n"
                
                for i, interaction in enumerate(interactions, 1):
                    severity = interaction["severity"]
                    
                    # Add appropriate emoji based on severity
                    if severity.lower() == "severe":
                        emoji = "🔴"
                    elif severity.lower() == "moderate":
                        emoji = "🟠"
                    else:
                        emoji = "🟡"
                        
                    response += f"{emoji} **{severity} interaction:** {interaction['description']}\n"
                    response += f"   Source: {interaction['source']}\n\n"
                
                # Add any management recommendations if available
                try:
                    literature_info = self.bio_llm.extract_ddi_from_literature(drug1, drug2)
                    if "interactions" in literature_info and literature_info["interactions"]:
                        management = literature_info["interactions"][0].get("management")
                        if management:
                            response += f"📝 **Management Recommendation:** {management}\n\n"
                except:
                    pass
                
                response += "⚕️ Please consult with a healthcare professional before taking these medications together."
                
                # Generate visualization
                try:
                    G, error = self.processor.generate_network(drug1, depth=1)
                    if G:
                        response += "\n\nA visualization of this interaction has been generated."
                    # In a real implementation, we would save the graph image and provide a link or display it
                except Exception as e:
                    pass  # Handle gracefully if visualization fails
                
                return response
        
        # Check if the user is asking for all interactions for a specific drug
        pattern = r"(what|show|list|tell).+?(interaction|interacts).+?(with|for|of)\s+(.+?)(?:\?|$)"
        match = re.search(pattern, message.lower())
        if match:
            drug_name = match.group(4).strip()
            canonical = self.db.search_drug(drug_name)
            
            if not canonical:
                return f"I couldn't find information about '{drug_name}' in our database."
            
            interactions, _ = self.db.get_all_interactions(canonical)
            
            if not interactions:
                return f"No known interactions were found for {canonical} in our database."
            
            response = f"**Known interactions for {canonical}:**\n\n"
            
            # Group by severity
            severe = []
            moderate = []
            mild = []
            
            for _, other_drug, desc, severity, source in interactions:
                if severity.lower() == "severe":
                    severe.append((other_drug, desc, source))
                elif severity.lower() == "moderate":
                    moderate.append((other_drug, desc, source))
                else:
                    mild.append((other_drug, desc, source))
            
            # Add severe interactions
            if severe:
                response += "🔴 **Severe interactions:**\n"
                for drug, desc, source in severe:
                    response += f"- **{drug}**: {desc} ({source})\n"
                response += "\n"
            
            # Add moderate interactions
            if moderate:
                response += "🟠 **Moderate interactions:**\n"
                for drug, desc, source in moderate:
                    response += f"- **{drug}**: {desc} ({source})\n"
                response += "\n"
            
            # Add mild interactions
            if mild:
                response += "🟡 **Mild interactions:**\n"
                for drug, desc, source in mild:
                    response += f"- **{drug}**: {desc} ({source})\n"
                response += "\n"
            
            response += "Please consult with a healthcare professional for personalized advice."
            
            return response
        
        # Check if the user is requesting a visualization
        if re.search(r'(visualize|visualization|graph|chart|network|diagram).+?(drug|interaction|medicine)', message.lower()):
            drug_match = re.search(r'(visualize|visualization|graph|chart|network|diagram).+?(for|of|between)\s+(.+?)(?:\?|$)', message.lower())
            
            if drug_match:
                drug_name = drug_match.group(3).strip()
                canonical = self.db.search_drug(drug_name)
                
                if not canonical:
                    return f"I couldn't find information about '{drug_name}' in our database."
                
                try:
                    G, error = self.processor.generate_network(canonical, depth=1)
                    if error:
                        return error
                    
                    return f"I've generated a network visualization for {canonical}'s interactions. The visualization shows connections to other drugs, with red edges indicating severe interactions, orange for moderate, and yellow for mild interactions."
                
                except Exception as e:
                    return f"Sorry, I encountered an error while generating the visualization: {str(e)}"
            
            else:
                try:
                    G, error = self.processor.generate_network()
                    if error:
                        return error
                    
                    return "I've generated a general drug interaction network visualization showing connections between several common drugs. Red edges indicate severe interactions, orange for moderate, and yellow for mild interactions."
                
                except Exception as e:
                    return f"Sorry, I encountered an error while generating the visualization: {str(e)}"
        
        # If not specifically about drugs
        return ("I'm a drug interaction assistant powered by biomedical language models. You can ask me about:\n\n"
               "1. Potential interactions between medications (e.g., 'Can I take aspirin and warfarin together?')\n"
               "2. Information about specific drugs (e.g., 'Tell me about metformin')\n"
               "3. Analysis of clinical notes (e.g., 'Analyze these clinical notes: [paste notes here]')\n"
               "4. Visualizations of drug interaction networks (e.g., 'Show me a visualization for warfarin')")
    
    def generate_visualization(self, drug_name=None, depth=1):
        """Generate a visualization of drug interactions"""
        G, error = self.processor.generate_network(drug_name, depth)
        
        if error:
            return None, error
        
        # Create a unique filename
        viz_id = str(uuid.uuid4())
        filename = f"static/visualizations/{viz_id}.png"
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        
        # Get positions
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
        
        # Save to file
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=150)
        plt.close()
        
        return filename, None
    
    def get_visualization_bytes(self, drug_name=None, depth=1):
        """Get visualization as bytes for web display"""
        G, error = self.processor.generate_network(drug_name, depth)
        
        if error:
            return None, error
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        
        # Get positions
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
        
        return buf, None 