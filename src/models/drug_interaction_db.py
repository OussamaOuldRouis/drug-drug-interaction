"""Drug Interaction Database with expanded information from PubMed abstracts."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DrugInteractionDatabase:
    def __init__(self):
        # Sample database with drug interactions
        # Format: (drug1, drug2, interaction_description, severity, source)
        self.interactions = [
            ("aspirin", "warfarin", "Increased risk of bleeding due to antiplatelet effects of aspirin combined with anticoagulant effects of warfarin. Both drugs affect different aspects of the coagulation cascade.", "Severe", "PubMed (PMID: 12345678)"),
            ("aspirin", "ibuprofen", "Ibuprofen may competitively inhibit the irreversible platelet inhibition induced by aspirin, potentially reducing cardiovascular benefits of aspirin.", "Moderate", "FDA warning (2006)"),
            ("simvastatin", "erythromycin", "Erythromycin inhibits CYP3A4 which metabolizes simvastatin, leading to increased plasma concentrations and risk of myopathy and rhabdomyolysis.", "Severe", "American College of Cardiology guidelines"),
            ("fluoxetine", "tramadol", "Both drugs increase serotonin levels, leading to potential serotonin syndrome characterized by agitation, hyperthermia, and neuromuscular abnormalities.", "Severe", "Case reports in Journal of Clinical Psychiatry"),
            ("amiodarone", "simvastatin", "Amiodarone inhibits CYP3A4 metabolism of simvastatin, increasing plasma levels and risk of myopathy.", "Severe", "FDA Drug Safety Communication"),
            ("warfarin", "vitamin k", "Vitamin K is a direct antagonist to warfarin's anticoagulant effects, acting as a cofactor for clotting factors II, VII, IX, and X.", "Moderate", "Pharmacotherapy journals"),
            ("ciprofloxacin", "theophylline", "Ciprofloxacin inhibits CYP1A2 which metabolizes theophylline, resulting in increased theophylline levels and potential toxicity.", "Moderate", "Clinical Pharmacokinetics studies"),
            ("metformin", "furosemide", "Furosemide may cause acute kidney injury in susceptible patients, which can increase metformin concentrations and risk of lactic acidosis.", "Moderate", "Case reports in Diabetes Care"),
            ("lithium", "nsaids", "NSAIDs reduce renal clearance of lithium by inhibiting prostaglandin synthesis, potentially causing lithium toxicity.", "Moderate", "American Psychiatric Association guidelines"),
            ("digoxin", "amiodarone", "Amiodarone increases digoxin levels through P-glycoprotein inhibition, requiring digoxin dose reduction of approximately 50%.", "Moderate", "Heart Rhythm Society guidelines"),
        ]
        
        # Additional drug information (generic and brand names)
        self.drug_aliases = {
            "aspirin": ["Bayer", "Ecotrin", "Bufferin", "acetylsalicylic acid", "asa"],
            "warfarin": ["Coumadin", "Jantoven"],
            "ibuprofen": ["Advil", "Motrin", "Nurofen"],
            "simvastatin": ["Zocor"],
            "erythromycin": ["E-Mycin", "Eryc", "Ery-Tab"],
            "fluoxetine": ["Prozac", "Sarafem"],
            "tramadol": ["Ultram", "ConZip"],
            "amiodarone": ["Pacerone", "Nexterone"],
            "vitamin k": ["phytonadione", "Mephyton"],
            "ciprofloxacin": ["Cipro"],
            "theophylline": ["Theo-24", "Elixophyllin"],
            "metformin": ["Glucophage", "Fortamet"],
            "furosemide": ["Lasix"],
            "lithium": ["Lithobid"],
            "nsaids": ["nonsteroidal anti-inflammatory drugs", "ibuprofen", "naproxen", "celecoxib"],
            "digoxin": ["Lanoxin"],
        }
        
        # Build a map from all possible names to canonical names
        self.name_to_canonical = {}
        for canonical, aliases in self.drug_aliases.items():
            self.name_to_canonical[canonical.lower()] = canonical
            for alias in aliases:
                self.name_to_canonical[alias.lower()] = canonical
                
        # Create embeddings for searching
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create sentence embeddings for all drug names and aliases for semantic search"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Collect all drug names and aliases
            all_names = []
            for canonical, aliases in self.drug_aliases.items():
                all_names.append(canonical)
                all_names.extend(aliases)
                
            # Create embeddings
            self.name_embeddings = self.model.encode(all_names)
            self.name_list = all_names
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            # Fallback to simple text matching if embeddings fail
            self.model = None
    
    def search_drug(self, query):
        """Search for a drug by name with fuzzy matching"""
        query = query.lower()
        
        # Direct match
        if query in self.name_to_canonical:
            return self.name_to_canonical[query]
        
        # Try semantic search if embeddings are available
        if hasattr(self, 'model') and self.model is not None:
            try:
                query_embedding = self.model.encode([query])
                similarities = cosine_similarity(query_embedding, self.name_embeddings)[0]
                
                # Get the index of the highest similarity
                best_match_idx = np.argmax(similarities)
                best_match_score = similarities[best_match_idx]
                
                # If similarity is high enough, return the match
                if best_match_score > 0.7:  # Threshold can be adjusted
                    best_match = self.name_list[best_match_idx]
                    return self.name_to_canonical.get(best_match.lower(), best_match)
            except:
                pass
        
        # Fallback to partial matching
        for name in self.name_to_canonical:
            if query in name or name in query:
                return self.name_to_canonical[name]
        
        return None
    
    def get_interactions(self, drug1, drug2):
        """Get interactions between two drugs"""
        canonical1 = self.search_drug(drug1)
        canonical2 = self.search_drug(drug2)
        
        if not canonical1 or not canonical2:
            missing = []
            if not canonical1:
                missing.append(drug1)
            if not canonical2:
                missing.append(drug2)
            return [], missing
        
        results = []
        for d1, d2, desc, severity, source in self.interactions:
            if (d1 == canonical1 and d2 == canonical2) or (d1 == canonical2 and d2 == canonical1):
                results.append((d1, d2, desc, severity, source))
        
        return results, []
    
    def get_all_interactions(self, drug):
        """Get all interactions for a specific drug"""
        canonical = self.search_drug(drug)
        if not canonical:
            return [], [drug]
        
        results = []
        for d1, d2, desc, severity, source in self.interactions:
            if d1 == canonical:
                results.append((d1, d2, desc, severity, source))
            elif d2 == canonical:
                results.append((d2, d1, desc, severity, source))
        
        return results, []
    
    def get_all_drugs(self):
        """Get a list of all drugs in the database"""
        return list(self.drug_aliases.keys()) 