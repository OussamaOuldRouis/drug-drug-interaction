"""Drug-Drug Interaction Processor for analyzing and processing drug interactions."""

import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

from .biomedical_llm import BiomedicalLLM
from .drug_interaction_db import DrugInteractionDatabase

class DDIProcessor:
    def __init__(self, db: DrugInteractionDatabase, bio_llm: BiomedicalLLM):
        self.db = db
        self.bio_llm = bio_llm
    
    def extract_drug_names(self, text):
        """Extract potential drug names from text using NLP techniques"""
        # In a real implementation, this would use advanced NLP
        # For now, we'll use a simple approach based on keywords and patterns
        
        # Clean and standardize text
        text = text.lower()
        
        # Common question patterns
        patterns = [
            r"can\s+i\s+take\s+(.+?)\s+(?:and|with|along\s+with)\s+(.+?)(?:\?|$)",
            r"is\s+it\s+safe\s+to\s+take\s+(.+?)\s+(?:and|with|along\s+with)\s+(.+?)(?:\?|$)",
            r"(?:interaction|interactions)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"(?:will|does|do)\s+(.+?)\s+(?:interact|interfere)\s+with\s+(.+?)(?:\?|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                drug1 = match.group(1).strip()
                drug2 = match.group(2).strip()
                return drug1, drug2
                
        # If no pattern matches, try to find drug names from the database
        words = text.split()
        potential_drugs = []
        
        for word in words:
            word = word.strip(".,?!()[]{}\"'")
            if self.db.search_drug(word):
                potential_drugs.append(word)
        
        if len(potential_drugs) >= 2:
            return potential_drugs[0], potential_drugs[1]
        
        return None, None
    
    def extract_drugs_from_clinical_notes(self, clinical_text):
        """Use BiomedLM to extract drugs from clinical notes"""
        try:
            # Use the biomedical LLM to extract drugs and interactions
            result = self.bio_llm.analyze_clinical_notes(clinical_text)
            
            # Return the extracted medications
            return result
        except Exception as e:
            print(f"Error extracting drugs from clinical notes: {e}")
            return {"medications": [], "potential_interactions": []}
    
    def process_query(self, query):
        """Process a natural language query about drug interactions"""
        drug1, drug2 = self.extract_drug_names(query)
        
        # If we couldn't extract drug names
        if not drug1 or not drug2:
            return {
                "status": "error",
                "message": "I couldn't identify the drugs in your question. Please specify the drugs clearly, for example: 'Can I take aspirin and warfarin together?'"
            }
        
        # Get drug interactions from database
        interactions, missing = self.db.get_interactions(drug1, drug2)
        
        # Try biomedical LLM for additional information, especially if not in database
        try:
            literature_info = self.bio_llm.extract_ddi_from_literature(drug1, drug2)
            if "interactions" in literature_info and literature_info["interactions"]:
                # Convert LLM information to the format used by the database
                for interaction in literature_info["interactions"]:
                    # Only add if we don't already have interactions from the database
                    if not interactions:
                        canonical1 = self.db.search_drug(drug1) or drug1
                        canonical2 = self.db.search_drug(drug2) or drug2
                        desc = interaction.get("description", f"Potential interaction between {drug1} and {drug2}")
                        severity = interaction.get("severity", "Unknown")
                        source = interaction.get("evidence", "Biomedical literature analysis")
                        
                        interactions.append((canonical1, canonical2, desc, severity, source))
                        
                # Clear missing drugs if LLM found information
                if missing and interactions:
                    missing = []
        except Exception as e:
            print(f"Error getting additional information: {e}")
        
        # If drugs weren't found
        if missing:
            return {
                "status": "not_found",
                "missing_drugs": missing,
                "message": f"I couldn't find information on the following drug(s): {', '.join(missing)}"
            }
        
        # Format the results
        canonical1 = self.db.search_drug(drug1) or drug1
        canonical2 = self.db.search_drug(drug2) or drug2
        
        if not interactions:
            return {
                "status": "no_interaction",
                "drugs": [canonical1, canonical2],
                "message": f"No known interactions were found between {canonical1} and {canonical2} in our database or medical literature. However, please consult with a healthcare professional for personalized advice."
            }
        
        # Format the interaction information
        interaction_details = []
        for d1, d2, desc, severity, source in interactions:
            interaction_details.append({
                "description": desc,
                "severity": severity,
                "source": source
            })
        
        return {
            "status": "found",
            "drugs": [canonical1, canonical2],
            "interactions": interaction_details
        }
    
    def get_drug_information(self, drug_name):
        """Get comprehensive information about a drug using biomedical LLM"""
        try:
            # First check if the drug exists in our database
            canonical = self.db.search_drug(drug_name)
            
            if not canonical:
                # If not in database, use just the provided name
                canonical = drug_name
            
            # Use biomedical LLM to get drug information
            drug_info = self.bio_llm.get_drug_information(canonical)
            
            # Add interactions from our database
            interactions, _ = self.db.get_all_interactions(canonical)
            interaction_drugs = []
            
            for d1, d2, _, severity, _ in interactions:
                other_drug = d2 if d1 == canonical else d1
                interaction_drugs.append(f"{other_drug} ({severity})")
            
            # Add to the drug information
            if interaction_drugs and "common_interactions" in drug_info:
                # Combine with LLM-provided interactions
                existing = drug_info["common_interactions"]
                if existing and existing[0] != "Information not available":
                    drug_info["common_interactions"] = list(set(existing + interaction_drugs))
                else:
                    drug_info["common_interactions"] = interaction_drugs
            
            return drug_info
            
        except Exception as e:
            print(f"Error getting drug information: {e}")
            return {
                "drug_name": drug_name,
                "drug_class": "Information not available",
                "mechanism": "Information not available",
                "indications": ["Information not available"],
                "side_effects": ["Information not available"],
                "common_interactions": ["Information not available"],
                "contraindications": ["Information not available"]
            }
    
    def generate_network(self, drug_name=None, depth=1):
        """
        Generate a network visualization of drug interactions
        If drug_name is provided, show interactions for that drug
        Otherwise, show a general interaction network
        """
        G = nx.Graph()
        
        # If a specific drug is provided
        if drug_name:
            canonical = self.db.search_drug(drug_name)
            if not canonical:
                return None, f"Drug '{drug_name}' not found"
            
            # Get interactions for this drug
            interactions, _ = self.db.get_all_interactions(canonical)
            
            # Add nodes and edges
            G.add_node(canonical, size=20, color='red')
            
            for drug1, drug2, desc, severity, _ in interactions:
                other_drug = drug2 if drug1 == canonical else drug1
                
                # Add nodes and edges
                if other_drug not in G:
                    G.add_node(other_drug, size=15, color='blue')
                
                # Set edge color based on severity
                if severity == "Severe":
                    edge_color = 'red'
                    weight = 3
                elif severity == "Moderate":
                    edge_color = 'orange'
                    weight = 2
                else:
                    edge_color = 'yellow'
                    weight = 1
                    
                G.add_edge(canonical, other_drug, color=edge_color, weight=weight, label=desc)
                
                # If depth > 1, add secondary interactions
                if depth > 1:
                    secondary_interactions, _ = self.db.get_all_interactions(other_drug)
                    for sec_d1, sec_d2, sec_desc, sec_severity, _ in secondary_interactions:
                        tertiary_drug = sec_d2 if sec_d1 == other_drug else sec_d1
                        
                        # Skip the original drug
                        if tertiary_drug == canonical:
                            continue
                        
                        if tertiary_drug not in G:
                            G.add_node(tertiary_drug, size=10, color='green')
                        
                        # Set edge color based on severity
                        if sec_severity == "Severe":
                            sec_edge_color = 'red'
                            sec_weight = 3
                        elif sec_severity == "Moderate":
                            sec_edge_color = 'orange'
                            sec_weight = 2
                        else:
                            sec_edge_color = 'yellow'
                            sec_weight = 1
                            
                        G.add_edge(other_drug, tertiary_drug, color=sec_edge_color, weight=sec_weight, label=sec_desc)
        else:
            # Create a general interaction network with common drugs
            sample_drugs = self.db.get_all_drugs()[:10]  # Limit to 10 drugs for clarity
            
            for drug in sample_drugs:
                G.add_node(drug, size=15, color='blue')
                
                interactions, _ = self.db.get_all_interactions(drug)
                for d1, d2, desc, severity, _ in interactions:
                    other_drug = d2 if d1 == drug else d1
                    
                    # Only add edges between drugs in our sample
                    if other_drug in sample_drugs:
                        # Set edge color based on severity
                        if severity == "Severe":
                            edge_color = 'red'
                            weight = 3
                        elif severity == "Moderate":
                            edge_color = 'orange'
                            weight = 2
                        else:
                            edge_color = 'yellow'
                            weight = 1
                            
                        G.add_edge(drug, other_drug, color=edge_color, weight=weight, label=desc)
        
        return G, None 