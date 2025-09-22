"""
Taxonomy Analyzer Module

Responsible for loading, parsing, and managing the taxonomy JSON file.
Provides access to all taxonomy rules, synonyms, and classification data.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import pickle


class TaxonomyAnalyzer:
    """Handles taxonomy file loading and provides rule-based analysis capabilities."""
    
    def __init__(self, config: Dict):
        """Initialize taxonomy analyzer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.taxonomy_data = None
        self.cache_file = "taxonomy_cache.pkl"
        
        # Parsed taxonomy structures
        self.tier1_types = []
        self.ambiguity_guardrails = []
        self.synonyms = []
        self.facet_rules = {}
        self.facets = set()
        
    def load_taxonomy(self) -> bool:
        """Load taxonomy from file or cache."""
        if self.config.get('cache_enabled', True) and Path(self.cache_file).exists():
            return self._load_from_cache()
        else:
            return self._load_from_file()
    
    def _load_from_cache(self) -> bool:
        """Load taxonomy from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.taxonomy_data = cached_data['taxonomy_data']
                self.tier1_types = cached_data['tier1_types']
                self.ambiguity_guardrails = cached_data['ambiguity_guardrails']
                self.synonyms = cached_data['synonyms']
                self.facet_rules = cached_data['facet_rules']
                self.facets = cached_data['facets']
            
            self.logger.info("Taxonomy loaded from cache")
            return True
        except Exception as e:
            self.logger.warning(f"Cache load failed: {e}. Loading from file...")
            return self._load_from_file()
    
    def _load_from_file(self) -> bool:
        """Load taxonomy from JSON file and parse it."""
        try:
            taxonomy_path = Path(self.config['file_path'])
            if not taxonomy_path.exists():
                self.logger.error(f"Taxonomy file not found: {taxonomy_path}")
                return False
            
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                self.taxonomy_data = json.load(f)
            
            self._parse_taxonomy()
            self._save_to_cache()
            
            self.logger.info("Taxonomy loaded from file and cached")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading taxonomy: {e}")
            return False
    
    def _parse_taxonomy(self):
        """Parse taxonomy data into structured formats."""
        # Parse Tier 1 Artifact Types
        if self.taxonomy_data and 'Tier 1 Artifact' in self.taxonomy_data:
            tier1_data = self.taxonomy_data['Tier 1 Artifact']
            self.tier1_types = tier1_data.get('Tier 1 Artifact Type', [])
            self.ambiguity_guardrails = tier1_data.get('Tier 1 Ambiguity guardrails', [])
            self.synonyms = tier1_data.get('Tier 1 synonyms', [])
        
        # Parse all rule sets by facet
        rule_sections = [
            'ProcessArea_Rules', 'SAPModule_Rules', 'ArtifactType_Rules',
            'LifecyclePhase_Rules', 'Status_Rules', 'Region_Rules',
            'SystemEnv_Rules', 'Confidentiality_Rules', 'GenericType_Rules',
            'ObjectCategory_Rules', 'Identifier_Extraction'
        ]
        
        if self.taxonomy_data:
            for section in rule_sections:
                if section in self.taxonomy_data:
                    facet_name = section.replace('_Rules', '').replace('_Extraction', '')
                    self.facet_rules[facet_name] = self.taxonomy_data[section]
                    self.facets.add(facet_name)
            
            # Add Domain Specific Artifact rules if present
            if 'Domain Specific Artifact' in self.taxonomy_data:
                self.facet_rules['Domain Specific Artifact'] = self.taxonomy_data['Domain Specific Artifact']
                self.facets.add('Domain Specific Artifact')
        
        self.logger.info(f"Parsed taxonomy with {len(self.facets)} facets and {len(self.tier1_types)} Tier 1 types")
    
    def _save_to_cache(self):
        """Save parsed taxonomy to cache file."""
        try:
            cache_data = {
                'taxonomy_data': self.taxonomy_data,
                'tier1_types': self.tier1_types,
                'ambiguity_guardrails': self.ambiguity_guardrails,
                'synonyms': self.synonyms,
                'facet_rules': self.facet_rules,
                'facets': self.facets
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def get_tier1_types(self) -> List[Dict]:
        """Get all Tier 1 artifact types."""
        return self.tier1_types
    
    def get_ambiguity_guardrails(self) -> List[Dict]:
        """Get ambiguity resolution rules."""
        return self.ambiguity_guardrails
    
    def get_synonyms(self) -> List[Dict]:
        """Get synonym mappings."""
        return self.synonyms
    
    def get_facet_rules(self, facet: str) -> List[Dict]:
        """Get rules for a specific facet."""
        return self.facet_rules.get(facet, [])
    
    def get_all_facets(self) -> Set[str]:
        """Get all available facets."""
        return self.facets
    
    def find_matching_rules(self, text: str, facet: str) -> List[Tuple[Dict, float]]:
        """Find rules that match the given text for a specific facet."""
        matching_rules = []
        rules = self.get_facet_rules(facet)
        
        for rule in rules:
            confidence = self._calculate_rule_confidence(text, rule)
            if confidence > 0:
                matching_rules.append((rule, confidence))
        
        # Sort by confidence descending
        matching_rules.sort(key=lambda x: x[1], reverse=True)
        return matching_rules
    
    def _calculate_rule_confidence(self, text: str, rule: Dict) -> float:
        """Calculate confidence score for a rule match."""
        try:
            # Get base weight from rule
            weight = float(rule.get('Weight', 0.5))
            
            # Check for regex patterns
            regex_pattern = rule.get('Examples / Regex', '')
            if regex_pattern and regex_pattern != '—':
                try:
                    if re.search(regex_pattern, text, re.IGNORECASE):
                        return weight
                except re.error:
                    pass
            
            # Check for keyword matches in signals
            signals = rule.get('If (signals)', '')
            if signals and signals != '—':
                keywords = [kw.strip().strip('"') for kw in signals.split(',')]
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        return weight * 0.8  # Slightly lower confidence for keyword match
            
            # Check anti-signals (reduce confidence if present)
            anti_signals = rule.get('Anti-signals', '')
            if anti_signals and anti_signals != '—':
                anti_keywords = [kw.strip() for kw in anti_signals.split('|')]
                for anti_kw in anti_keywords:
                    if anti_kw.lower() in text.lower():
                        return 0  # Rule doesn't apply if anti-signal is present
            
            return 0
            
        except Exception as e:
            self.logger.debug(f"Error calculating rule confidence: {e}")
            return 0
    
    def resolve_ambiguity(self, text: str, candidate_tags: List[str]) -> Optional[str]:
        """Use ambiguity guardrails to resolve conflicts between candidate tags."""
        for guardrail in self.ambiguity_guardrails:
            target_tag = guardrail.get('Tag as →', '')
            
            if target_tag in candidate_tags:
                # Check if signals for this guardrail are present
                signals = guardrail.get('Use when… (primary intent / signals)', '')
                detection_cues = guardrail.get('Detection cues / keywords', '')
                
                signal_match = any(signal.strip().lower() in text.lower() 
                                 for signal in signals.split(',') if signal.strip())
                cue_match = any(cue.strip().strip('"').lower() in text.lower() 
                               for cue in detection_cues.split(',') if cue.strip())
                
                if signal_match or cue_match:
                    return target_tag
        
        return None
    
    def get_synonym_mapping(self, text: str) -> Optional[str]:
        """Find the canonical form for synonyms/aliases."""
        for synonym_entry in self.synonyms:
            generic_type = synonym_entry.get('Generic Document Type', '')
            aliases = synonym_entry.get('Common altLabels / aliases', '')
            
            if aliases:
                alias_list = [alias.strip() for alias in aliases.split(',')]
                for alias in alias_list:
                    if alias.lower() in text.lower():
                        return generic_type
        
        return None