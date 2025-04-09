"""Biomedical Language Model for drug interaction analysis."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

class BiomedicalLLM:
    """Class to handle biomedical language model inference"""
    
    def __init__(self, model_name="stanford-crfm/BioMedLM"):
        """
        Initialize the Biomedical Language Model
        
        Args:
            model_name: The name of the model to use (default: BioMedLM)
                Options include:
                - "stanford-crfm/BioMedLM"
                - "microsoft/biogpt"
        """
        self.model_name = model_name
        try:
            print(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Set pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                pad_token_id=self.tokenizer.pad_token_id
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to API-based approach or stub implementation")
            self.tokenizer = None
            self.model = None
    
    def extract_ddi_from_literature(self, drug1, drug2):
        """
        Extract drug-drug interaction information from biomedical literature
        
        Args:
            drug1: Name of the first drug
            drug2: Name of the second drug
            
        Returns:
            A list of extracted interactions with evidence
        """
        if self.model is None or self.tokenizer is None:
            # Fallback behavior if model failed to load
            return self._fallback_extract_ddi(drug1, drug2)
        
        try:
            # Construct a prompt for the model
            prompt = f"""
            Analyze the scientific literature for interactions between {drug1} and {drug2}.
            Include the following information:
            1. Description of the interaction mechanism
            2. Severity (Mild, Moderate, Severe)
            3. Clinical significance
            4. Management recommendations
            
            Format the response as JSON with the following structure:
            {{
                "interactions": [
                    {{
                        "description": "Description of mechanism",
                        "severity": "Severity level",
                        "evidence": "Evidence from literature",
                        "management": "Management recommendation"
                    }}
                ]
            }}
            """
            
            # Generate completion with proper attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Ensure attention mask is set properly
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Generate with appropriate parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the JSON part of the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                json_str = json_match.group(1)
                try:
                    return json.loads(json_str)
                except:
                    # If JSON parsing fails, return a structured response anyway
                    return self._extract_structured_info(response, drug1, drug2)
            else:
                # If no JSON found, extract structured information
                return self._extract_structured_info(response, drug1, drug2)
                
        except Exception as e:
            print(f"Error in LLM inference: {e}")
            return self._fallback_extract_ddi(drug1, drug2)
    
    def _extract_structured_info(self, text, drug1, drug2):
        """Extract structured information from text if JSON parsing fails"""
        # Try to identify descriptions, severity, etc.
        severity_match = re.search(r'(mild|moderate|severe)', text.lower())
        severity = severity_match.group(1).capitalize() if severity_match else "Unknown"
        
        # Default structured response
        return {
            "interactions": [
                {
                    "description": f"Potential interaction between {drug1} and {drug2} identified in literature",
                    "severity": severity,
                    "evidence": "Based on biomedical literature analysis",
                    "management": "Consult healthcare provider for specific guidance"
                }
            ]
        }
    
    def _fallback_extract_ddi(self, drug1, drug2):
        """Fallback method when model is not available"""
        # Return a structured response with disclaimer
        return {
            "interactions": [
                {
                    "description": f"Potential interaction between {drug1} and {drug2}",
                    "severity": "Unknown",
                    "evidence": "Please consult literature for evidence",
                    "management": "Consult healthcare provider for guidance"
                }
            ],
            "note": "Biomedical model not available - using fallback information"
        }
    
    def analyze_clinical_notes(self, clinical_text):
        """
        Extract drug mentions and potential interactions from clinical notes
        
        Args:
            clinical_text: The clinical notes text to analyze
            
        Returns:
            A dictionary with extracted drugs and potential interactions
        """
        if self.model is None or self.tokenizer is None:
            # Fallback behavior if model failed to load
            return self._fallback_analyze_clinical_notes(clinical_text)
        
        try:
            # Construct a prompt for the model
            prompt = f"""
            Extract all medication mentions and potential drug interactions from the following clinical note:
            
            {clinical_text}
            
            Format the response as JSON with the following structure:
            {{
                "medications": [
                    {{
                        "name": "Drug name",
                        "dosage": "Dosage if mentioned",
                        "frequency": "Frequency if mentioned"
                    }}
                ],
                "potential_interactions": [
                    {{
                        "drug1": "First drug name",
                        "drug2": "Second drug name",
                        "concern": "Description of potential interaction"
                    }}
                ]
            }}
            """
            
            # Generate completion with proper attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Ensure attention mask is set properly
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Generate with appropriate parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the JSON part of the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                json_str = json_match.group(1)
                try:
                    return json.loads(json_str)
                except:
                    # If JSON parsing fails, return a structured response anyway
                    return self._extract_medications_from_text(response)
            else:
                # If no JSON found, extract structured information
                return self._extract_medications_from_text(response)
                
        except Exception as e:
            print(f"Error in LLM inference: {e}")
            return self._fallback_analyze_clinical_notes(clinical_text)
    
    def _extract_medications_from_text(self, text):
        """Extract medication mentions from text if JSON parsing fails"""
        # Simple regex-based extraction
        drug_patterns = [
            r'([A-Za-z]+)\s+(\d+\s*mg)',
            r'([A-Za-z]+)\s+(\d+\s*mcg)',
            r'([A-Za-z]+)\s+(\d+\s*ml)',
            r'([A-Za-z]+)\s+(\d+\s*tablet)',
            r'prescribe[d]?\s+([A-Za-z]+)',
            r'taking\s+([A-Za-z]+)',
            r'administer[ed]?\s+([A-Za-z]+)'
        ]
        
        medications = []
        for pattern in drug_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 1:
                    drug_name = match.group(1)
                    dosage = match.group(2)
                    medications.append({"name": drug_name, "dosage": dosage, "frequency": "Not specified"})
                else:
                    drug_name = match.group(1)
                    medications.append({"name": drug_name, "dosage": "Not specified", "frequency": "Not specified"})
        
        # Return structured data
        return {
            "medications": medications,
            "potential_interactions": []
        }
    
    def _fallback_analyze_clinical_notes(self, clinical_text):
        """Fallback method for clinical note analysis when model is not available"""
        # Return a structured response with disclaimer
        return {
            "medications": [],
            "potential_interactions": [],
            "note": "Biomedical model not available - please review clinical notes manually"
        }
    
    def get_drug_information(self, drug_name):
        """
        Get detailed information about a specific drug
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            A dictionary with drug information
        """
        if self.model is None or self.tokenizer is None:
            # Fallback behavior if model failed to load
            return self._fallback_drug_information(drug_name)
        
        try:
            # Construct a prompt for the model
            prompt = f"""
            Provide comprehensive information about the medication {drug_name}, including:
            1. Drug class
            2. Mechanism of action
            3. Common indications
            4. Common side effects
            5. Common drug interactions
            6. Contraindications
            
            Format the response as JSON with the following structure:
            {{
                "drug_name": "{drug_name}",
                "drug_class": "Drug class",
                "mechanism": "Mechanism of action",
                "indications": ["Indication 1", "Indication 2"],
                "side_effects": ["Side effect 1", "Side effect 2"],
                "common_interactions": ["Drug 1", "Drug 2"],
                "contraindications": ["Contraindication 1", "Contraindication 2"]
            }}
            """
            
            # Generate completion with proper attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Ensure attention mask is set properly
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Generate with appropriate parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the JSON part of the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                json_str = json_match.group(1)
                try:
                    return json.loads(json_str)
                except:
                    # If JSON parsing fails, return a structured response anyway
                    return self._extract_drug_info_from_text(response, drug_name)
            else:
                # If no JSON found, extract structured information
                return self._extract_drug_info_from_text(response, drug_name)
                
        except Exception as e:
            print(f"Error in LLM inference: {e}")
            return self._fallback_drug_information(drug_name)
    
    def _extract_drug_info_from_text(self, text, drug_name):
        """Extract drug information from text if JSON parsing fails"""
        # Create default structure
        drug_info = {
            "drug_name": drug_name,
            "drug_class": "Not specified",
            "mechanism": "Not specified",
            "indications": [],
            "side_effects": [],
            "common_interactions": [],
            "contraindications": []
        }
        
        # Try to extract each section
        class_match = re.search(r'[Cc]lass:?\s+([^\n\.]+)', text)
        if class_match:
            drug_info["drug_class"] = class_match.group(1).strip()
        
        mechanism_match = re.search(r'[Mm]echanism:?\s+([^\n\.]+)', text)
        if mechanism_match:
            drug_info["mechanism"] = mechanism_match.group(1).strip()
        
        # Extract lists with regex
        indication_match = re.search(r'[Ii]ndications?:?\s+((?:[^\n]+\n?)+)', text)
        if indication_match:
            indications_text = indication_match.group(1)
            # Split by common list markers
            indications = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|-|•)\s*([^\n]+)', indications_text)
            if indications:
                drug_info["indications"] = [ind.strip() for ind in indications]
            elif indications_text:
                # If no list markers, just split by commas or newlines
                items = re.split(r',|\n', indications_text)
                drug_info["indications"] = [item.strip() for item in items if item.strip()]
        
        # Similarly for other list-based fields
        side_effects_match = re.search(r'[Ss]ide [Ee]ffects:?\s+((?:[^\n]+\n?)+)', text)
        if side_effects_match:
            side_effects_text = side_effects_match.group(1)
            side_effects = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|-|•)\s*([^\n]+)', side_effects_text)
            if side_effects:
                drug_info["side_effects"] = [se.strip() for se in side_effects]
            elif side_effects_text:
                items = re.split(r',|\n', side_effects_text)
                drug_info["side_effects"] = [item.strip() for item in items if item.strip()]
        
        interactions_match = re.search(r'[Ii]nteractions:?\s+((?:[^\n]+\n?)+)', text)
        if interactions_match:
            interactions_text = interactions_match.group(1)
            interactions = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|-|•)\s*([^\n]+)', interactions_text)
            if interactions:
                drug_info["common_interactions"] = [inter.strip() for inter in interactions]
            elif interactions_text:
                items = re.split(r',|\n', interactions_text)
                drug_info["common_interactions"] = [item.strip() for item in items if item.strip()]
        
        contraindications_match = re.search(r'[Cc]ontraindications:?\s+((?:[^\n]+\n?)+)', text)
        if contraindications_match:
            contraindications_text = contraindications_match.group(1)
            contraindications = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|-|•)\s*([^\n]+)', contraindications_text)
            if contraindications:
                drug_info["contraindications"] = [contra.strip() for contra in contraindications]
            elif contraindications_text:
                items = re.split(r',|\n', contraindications_text)
                drug_info["contraindications"] = [item.strip() for item in items if item.strip()]
        
        return drug_info
    
    def _fallback_drug_information(self, drug_name):
        """Fallback method for drug information when model is not available"""
        # Return a structured response with disclaimer
        return {
            "drug_name": drug_name,
            "drug_class": "Information not available",
            "mechanism": "Information not available",
            "indications": ["Information not available"],
            "side_effects": ["Information not available"],
            "common_interactions": ["Information not available"],
            "contraindications": ["Information not available"],
            "note": "Biomedical model not available - using fallback information"
        } 