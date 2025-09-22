"""
Tag Assigner Module

Assigns the most appropriate tags from each of the 13 taxonomy facets to documents.
Ensures each document gets exactly one tag from each required facet.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict


class TagAssigner:
    """Assigns taxonomy tags to documents using rule-based and LLM analysis."""
    
    # The 13 required facets in the exact order specified
    REQUIRED_FACETS = [
        'Tier 1 Artifact Type',
        'Domain Specific Artifact', 
        'ArtifactType',
        'ProcessArea',
        'SAPModule',
        'ObjectCategory',
        'LifecyclePhase',
        'Status',
        'Region',
        'SystemEnv',
        'Confidentiality',
        'GenericType',
        'Identifier'
    ]
    
    def __init__(self, taxonomy_analyzer, confidence_scorer):
        """Initialize tag assigner with taxonomy analyzer and confidence scorer."""
        self.taxonomy_analyzer = taxonomy_analyzer
        self.confidence_scorer = confidence_scorer
        self.logger = logging.getLogger(__name__)
        
        # Cache for facet mappings
        self._facet_mapping_cache = {}
        self._initialize_facet_mappings()
    
    def _initialize_facet_mappings(self):
        """Initialize mappings between required facets and taxonomy data."""
        # Map required facets to taxonomy data structure names
        self.facet_mappings = {
            'Tier 1 Artifact Type': 'Tier 1 Artifact',
            'Domain Specific Artifact': 'Domain Specific Artifact',
            'ArtifactType': 'ArtifactType',
            'ProcessArea': 'ProcessArea',
            'SAPModule': 'SAPModule', 
            'ObjectCategory': 'ObjectCategory',
            'LifecyclePhase': 'LifecyclePhase',
            'Status': 'Status',
            'Region': 'Region',
            'SystemEnv': 'SystemEnv',
            'Confidentiality': 'Confidentiality',
            'GenericType': 'GenericType',
            'Identifier': 'Identifier'
        }
    
    def assign_tags(self, document_content: str, intent_analysis: Dict, 
                   filename: str) -> Dict[str, str]:
        """Assign tags for all 13 facets to a document."""
        # Store document content for confidence scorer
        self.confidence_scorer._current_document_content = document_content
        
        tag_results = {}
        all_candidates = {}
        
        # Process each required facet
        for required_facet in self.REQUIRED_FACETS:
            try:
                candidates = self._get_facet_candidates(
                    document_content, intent_analysis, required_facet
                )
                
                all_candidates[required_facet] = candidates
                
                # Select best tag for this facet
                best_tag = self._select_best_tag(candidates, required_facet)
                tag_results[required_facet] = best_tag
                
                self.logger.debug(f"Assigned {required_facet}: {best_tag}")
                
            except Exception as e:
                self.logger.error(f"Error processing facet {required_facet}: {e}")
                tag_results[required_facet] = "Unknown"
        
        # Resolve any conflicts between related facets
        tag_results = self._resolve_cross_facet_conflicts(tag_results, document_content)
        
        # Ensure all required facets have values
        tag_results = self._ensure_all_facets_populated(tag_results)
        
        self.logger.info(f"Tag assignment complete for {filename}")
        
        return tag_results
    
    def _get_facet_candidates(self, document_content: str, intent_analysis: Dict, 
                            facet_name: str) -> List[Dict]:
        """Get candidate classifications for a specific facet."""
        candidates = []
        
        # Get taxonomy facet name
        taxonomy_facet = self.facet_mappings.get(facet_name, facet_name)
        
        try:
            # Special handling for Tier 1 Artifact Type
            if facet_name == 'Tier 1 Artifact Type':
                candidates = self._get_tier1_candidates(document_content, intent_analysis)
            
            # Special handling for Domain Specific Artifact
            elif facet_name == 'Domain Specific Artifact':
                candidates = self._get_domain_specific_candidates(document_content, intent_analysis)
            
            # Rule-based facets
            else:
                rule_matches = self.taxonomy_analyzer.find_matching_rules(document_content, taxonomy_facet)
                
                if rule_matches:
                    candidates = self.confidence_scorer.score_facet_candidates(
                        document_content, facet_name, rule_matches
                    )
                
                # Enhance with LLM if we have few or low-confidence matches
                if len(candidates) < 2 or (candidates and candidates[0]['confidence'] < 0.5):
                    candidates = self._enhance_candidates_with_llm(
                        candidates, document_content, intent_analysis, facet_name
                    )
            
            # Apply confidence threshold
            candidates = [c for c in candidates if c['confidence'] >= 0.1]
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error getting candidates for {facet_name}: {e}")
            return []
    
    def _get_tier1_candidates(self, document_content: str, intent_analysis: Dict) -> List[Dict]:
        """Get candidates for Tier 1 Artifact Type."""
        candidates = []
        tier1_types = self.taxonomy_analyzer.get_tier1_types()
        
        for tier1_type in tier1_types:
            confidence = self._calculate_tier1_confidence(document_content, tier1_type, intent_analysis)
            
            if confidence > 0.1:
                candidate = {
                    'facet': 'Tier 1 Artifact Type',
                    'classification': tier1_type['Acronym'],
                    'full_name': tier1_type['GenericType'],
                    'confidence': confidence,
                    'reasoning': f"Matched {tier1_type['GenericType']} pattern"
                }
                candidates.append(candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return candidates
    
    def _calculate_tier1_confidence(self, document_content: str, tier1_type: Dict, 
                                   intent_analysis: Dict) -> float:
        """Calculate confidence for Tier 1 artifact type matching."""
        confidence = 0.0
        text_lower = document_content.lower()
        
        # Check for exact type name match
        generic_type = tier1_type.get('GenericType', '').lower()
        if generic_type in text_lower:
            confidence += 0.6
        
        # Check for acronym match
        acronym = tier1_type.get('Acronym', '').lower()
        if acronym and acronym in text_lower:
            confidence += 0.4
        
        # Check aliases
        aliases = tier1_type.get('Common altLabels / aliases', '')
        if aliases:
            for alias in aliases.split(','):
                alias = alias.strip().lower()
                if alias and alias in text_lower:
                    confidence += 0.3
                    break
        
        # Use LLM analysis for additional context
        llm_confidence = self._get_llm_tier1_confidence(tier1_type, intent_analysis)
        confidence = max(confidence, llm_confidence)
        
        return min(1.0, confidence)
    
    def _get_llm_tier1_confidence(self, tier1_type: Dict, intent_analysis: Dict) -> float:
        """Get LLM-based confidence for Tier 1 type."""
        # Simple heuristic based on intent analysis
        generic_type = tier1_type.get('GenericType', '').lower()
        
        indicators = [ind.lower() for ind in intent_analysis.get('document_type_indicators', [])]
        characteristics = [char.lower() for char in intent_analysis.get('key_characteristics', [])]
        
        all_indicators = indicators + characteristics
        
        for indicator in all_indicators:
            if generic_type in indicator or any(word in indicator for word in generic_type.split()):
                return 0.5
        
        return 0.0
    
    def _get_domain_specific_candidates(self, document_content: str, intent_analysis: Dict) -> List[Dict]:
        """Get candidates for Domain Specific Artifact."""
        candidates = []
        
        # Get domain-specific rules from taxonomy
        domain_rules = self.taxonomy_analyzer.taxonomy_data.get('Domain Specific Artifact', {})
        
        if 'Domain Specific Synonyms' in domain_rules:
            for domain_item in domain_rules['Domain Specific Synonyms']:
                confidence = self._calculate_domain_confidence(document_content, domain_item)
                
                if confidence > 0.1:
                    candidate = {
                        'facet': 'Domain Specific Artifact',
                        'classification': domain_item.get('ArtifactType', 'Unknown'),
                        'confidence': confidence,
                        'reasoning': f"Domain-specific pattern match"
                    }
                    candidates.append(candidate)
        
        # If no domain-specific matches, return a default
        if not candidates:
            candidates.append({
                'facet': 'Domain Specific Artifact',
                'classification': 'Generic',
                'confidence': 0.2,
                'reasoning': 'Default classification'
            })
        
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates
    
    def _calculate_domain_confidence(self, document_content: str, domain_item: Dict) -> float:
        """Calculate confidence for domain-specific artifact matching."""
        confidence = 0.0
        text_lower = document_content.lower()
        
        # Check artifact type
        artifact_type = domain_item.get('ArtifactType', '').lower()
        if artifact_type and artifact_type in text_lower:
            confidence += 0.5
        
        # Check synonyms
        synonyms = domain_item.get('Synonyms', '')
        if synonyms:
            for synonym in synonyms.split(','):
                synonym = synonym.strip().lower()
                if synonym and synonym in text_lower:
                    confidence += 0.3
                    break
        
        return min(1.0, confidence)
    
    def _enhance_candidates_with_llm(self, existing_candidates: List[Dict], 
                                   document_content: str, intent_analysis: Dict, 
                                   facet_name: str) -> List[Dict]:
        """Enhance candidates using LLM analysis."""
        try:
            # For now, return existing candidates
            # Could be enhanced with LLM classification in future
            return existing_candidates
            
        except Exception as e:
            self.logger.error(f"Error enhancing candidates with LLM: {e}")
            return existing_candidates
    
    def _select_best_tag(self, candidates: List[Dict], facet_name: str) -> str:
        """Select the best tag from candidates for a facet."""
        if not candidates:
            return self._get_default_tag(facet_name)
        
        # Return highest confidence candidate
        best_candidate = candidates[0]
        
        # Additional validation
        if best_candidate['confidence'] < 0.2:
            return self._get_default_tag(facet_name)
        
        return best_candidate['classification']
    
    def _get_default_tag(self, facet_name: str) -> str:
        """Get default tag for a facet when no matches are found."""
        defaults = {
            'Tier 1 Artifact Type': 'DOC',  # Generic document
            'Domain Specific Artifact': 'General',
            'ArtifactType': 'General',
            'ProcessArea': 'General',
            'SAPModule': 'Cross-Module',
            'ObjectCategory': 'Document',
            'LifecyclePhase': 'Active',
            'Status': 'Draft',
            'Region': 'Global',
            'SystemEnv': 'General',
            'Confidentiality': 'Internal',
            'GenericType': 'Document',
            'Identifier': 'N/A'
        }
        
        return defaults.get(facet_name, 'Unknown')
    
    def _resolve_cross_facet_conflicts(self, tag_results: Dict[str, str], 
                                     document_content: str) -> Dict[str, str]:
        """Resolve conflicts between related facets."""
        # Example: If Tier 1 is "Policy", GenericType should align
        tier1 = tag_results.get('Tier 1 Artifact Type', '')
        generic = tag_results.get('GenericType', '')
        
        # Use ambiguity guardrails from taxonomy
        if tier1 and generic:
            resolved_tag = self.taxonomy_analyzer.resolve_ambiguity(
                document_content, [tier1, generic]
            )
            if resolved_tag:
                # Update the appropriate facet based on resolution
                if resolved_tag in [item['Acronym'] for item in self.taxonomy_analyzer.get_tier1_types()]:
                    tag_results['Tier 1 Artifact Type'] = resolved_tag
                else:
                    tag_results['GenericType'] = resolved_tag
        
        return tag_results
    
    def _ensure_all_facets_populated(self, tag_results: Dict[str, str]) -> Dict[str, str]:
        """Ensure all 13 required facets have values."""
        for facet in self.REQUIRED_FACETS:
            if facet not in tag_results or not tag_results[facet]:
                tag_results[facet] = self._get_default_tag(facet)
                self.logger.warning(f"Using default tag for {facet}: {tag_results[facet]}")
        
        return tag_results
    
    def get_assignment_summary(self, tag_results: Dict[str, str]) -> Dict[str, Any]:
        """Generate a summary of tag assignments."""
        return {
            'total_facets': len(self.REQUIRED_FACETS),
            'assigned_facets': len([v for v in tag_results.values() if v != 'Unknown']),
            'default_assignments': len([v for v in tag_results.values() if v in ['Unknown', 'General', 'N/A']]),
            'facets_with_values': list(tag_results.keys())
        }