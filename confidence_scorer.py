"""
Confidence Scorer Module

Calculates confidence scores for taxonomy rule matches using various algorithms.
Combines rule-based scoring with LLM-enhanced contextual analysis.
"""

import re
import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class ConfidenceScorer:
    """Calculates and manages confidence scores for taxonomy classifications."""
    
    def __init__(self, taxonomy_analyzer):
        """Initialize confidence scorer with taxonomy analyzer."""
        self.taxonomy_analyzer = taxonomy_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Scoring parameters
        self.base_confidence_threshold = 0.3
        self.keyword_match_boost = 0.1
        self.regex_match_boost = 0.2
        self.anti_signal_penalty = -0.5
        self.length_normalization_factor = 1000  # characters
    
    def calculate_rule_confidence(self, document_content: str, rule: Dict, 
                                facet_name: str) -> float:
        """Calculate confidence score for a specific rule match."""
        try:
            base_weight = float(rule.get('Weight', 0.5))
            confidence = 0.0
            
            # Regex pattern matching
            regex_confidence = self._calculate_regex_confidence(document_content, rule)
            
            # Keyword signal matching
            signal_confidence = self._calculate_signal_confidence(document_content, rule)
            
            # Anti-signal penalty
            anti_signal_penalty = self._calculate_anti_signal_penalty(document_content, rule)
            
            # Combine scores
            confidence = max(regex_confidence, signal_confidence) * base_weight
            confidence += anti_signal_penalty
            
            # Apply document length normalization
            confidence = self._normalize_for_document_length(confidence, document_content)
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            self.logger.debug(f"Rule {rule.get('Rule ID', 'unknown')} confidence: {confidence:.3f}")
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating rule confidence: {e}")
            return 0.0
    
    def _calculate_regex_confidence(self, text: str, rule: Dict) -> float:
        """Calculate confidence based on regex pattern matches."""
        regex_pattern = rule.get('Examples / Regex', '')
        if not regex_pattern or regex_pattern in ['—', '—', 'Default fallback']:
            return 0.0
        
        try:
            matches = re.findall(regex_pattern, text, re.IGNORECASE)
            if matches:
                # More matches = higher confidence (with diminishing returns)
                match_count = len(matches)
                confidence = min(1.0, 0.3 + (match_count * 0.1))
                return confidence + self.regex_match_boost
            
        except re.error as e:
            self.logger.debug(f"Invalid regex pattern: {regex_pattern}, error: {e}")
        
        return 0.0
    
    def _calculate_signal_confidence(self, text: str, rule: Dict) -> float:
        """Calculate confidence based on signal keyword matches."""
        signals = rule.get('If (signals)', '')
        if not signals or signals in ['—', '—']:
            return 0.0
        
        text_lower = text.lower()
        matched_signals = 0
        total_signals = 0
        
        # Parse signals (comma or pipe separated)
        signal_list = []
        for separator in [',', '|']:
            if separator in signals:
                signal_list = [s.strip().strip('"').strip("'") for s in signals.split(separator)]
                break
        
        if not signal_list:
            signal_list = [signals.strip().strip('"').strip("'")]
        
        for signal in signal_list:
            if signal and signal.lower() in text_lower:
                matched_signals += 1
            total_signals += 1
        
        if total_signals == 0:
            return 0.0
        
        # Calculate match ratio with boost
        match_ratio = matched_signals / total_signals
        confidence = match_ratio * 0.6  # Base confidence for signal matches
        
        if matched_signals > 0:
            confidence += self.keyword_match_boost
        
        return confidence
    
    def _calculate_anti_signal_penalty(self, text: str, rule: Dict) -> float:
        """Calculate penalty for anti-signal presence."""
        anti_signals = rule.get('Anti-signals', '')
        if not anti_signals or anti_signals in ['—', '—']:
            return 0.0
        
        text_lower = text.lower()
        
        # Parse anti-signals (pipe separated typically)
        anti_signal_list = [s.strip() for s in anti_signals.split('|')]
        
        for anti_signal in anti_signal_list:
            if anti_signal and anti_signal.lower() in text_lower:
                return self.anti_signal_penalty
        
        return 0.0
    
    def _normalize_for_document_length(self, confidence: float, document_content: str) -> float:
        """Normalize confidence score based on document length."""
        doc_length = len(document_content)
        
        # Longer documents might have incidental matches, slightly reduce confidence
        if doc_length > self.length_normalization_factor:
            length_factor = math.log(doc_length / self.length_normalization_factor) * 0.05
            confidence = confidence * (1 - length_factor)
        
        return max(0.0, confidence)
    
    def score_facet_candidates(self, document_content: str, facet_name: str,
                             rule_matches: List[Tuple[Dict, float]]) -> List[Dict]:
        """Score all candidate classifications for a facet."""
        scored_candidates = []
        
        for rule, base_confidence in rule_matches:
            candidate = {
                'facet': facet_name,
                'classification': rule.get('Set (value)', 'Unknown'),
                'rule_id': rule.get('Rule ID', ''),
                'confidence': base_confidence,
                'rule': rule,
                'reasoning': self._generate_reasoning(rule, base_confidence)
            }
            scored_candidates.append(candidate)
        
        # Sort by confidence descending
        scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return scored_candidates
    
    def _generate_reasoning(self, rule: Dict, confidence: float) -> str:
        """Generate human-readable reasoning for the confidence score."""
        reasoning_parts = []
        
        rule_id = rule.get('Rule ID', 'Unknown')
        reasoning_parts.append(f"Rule {rule_id}")
        
        if confidence >= 0.7:
            reasoning_parts.append("high confidence match")
        elif confidence >= 0.5:
            reasoning_parts.append("medium confidence match")
        else:
            reasoning_parts.append("low confidence match")
        
        signals = rule.get('If (signals)', '')
        if signals and signals != '—':
            reasoning_parts.append(f"based on signals: {signals[:50]}...")
        
        return " - ".join(reasoning_parts)
    
    def resolve_conflicts(self, candidates_by_facet: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Resolve conflicts between overlapping classifications."""
        resolved = {}
        
        for facet_name, candidates in candidates_by_facet.items():
            if not candidates:
                resolved[facet_name] = {
                    'classification': 'Unknown',
                    'confidence': 0.0,
                    'reasoning': 'No matching rules found'
                }
                continue
            
            # Get highest confidence candidate
            best_candidate = candidates[0]
            
            # Check for ambiguity resolution if multiple high-confidence candidates
            if len(candidates) > 1 and candidates[1]['confidence'] > 0.5:
                # Use taxonomy ambiguity guardrails
                document_content = getattr(self, '_current_document_content', '')
                candidate_tags = [c['classification'] for c in candidates[:3]]
                
                resolved_tag = self.taxonomy_analyzer.resolve_ambiguity(
                    document_content, candidate_tags
                )
                
                if resolved_tag:
                    # Find the candidate that matches the resolved tag
                    for candidate in candidates:
                        if candidate['classification'] == resolved_tag:
                            best_candidate = candidate.copy()
                            best_candidate['confidence'] = min(1.0, best_candidate['confidence'] + 0.1)
                            best_candidate['reasoning'] += " (ambiguity resolved)"
                            break
            
            resolved[facet_name] = {
                'classification': best_candidate['classification'],
                'confidence': best_candidate['confidence'],
                'reasoning': best_candidate['reasoning'],
                'rule_id': best_candidate.get('rule_id', '')
            }
        
        return resolved
    
    def enhance_with_llm_confidence(self, candidates: List[Dict], 
                                  llm_analysis: Dict) -> List[Dict]:
        """Enhance rule-based confidence with LLM insights."""
        try:
            enhanced_candidates = []
            
            for candidate in candidates:
                enhanced = candidate.copy()
                
                # Get LLM confidence for this classification
                llm_confidence = self._get_llm_confidence_for_classification(
                    candidate['classification'], llm_analysis
                )
                
                # Combine rule-based and LLM confidence
                combined_confidence = self._combine_confidence_scores(
                    candidate['confidence'], llm_confidence
                )
                
                enhanced['confidence'] = combined_confidence
                enhanced['llm_confidence'] = llm_confidence
                enhanced['original_rule_confidence'] = candidate['confidence']
                
                enhanced_candidates.append(enhanced)
            
            # Re-sort by enhanced confidence
            enhanced_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            return enhanced_candidates
            
        except Exception as e:
            self.logger.error(f"Error enhancing with LLM confidence: {e}")
            return candidates
    
    def _get_llm_confidence_for_classification(self, classification: str, 
                                             llm_analysis: Dict) -> float:
        """Extract confidence for a specific classification from LLM analysis."""
        # Look for relevant indicators in LLM analysis
        indicators = llm_analysis.get('document_type_indicators', [])
        characteristics = llm_analysis.get('key_characteristics', [])
        
        # Simple keyword matching for now - could be more sophisticated
        all_text = ' '.join(indicators + characteristics).lower()
        classification_lower = classification.lower()
        
        if classification_lower in all_text:
            return 0.7
        
        # Check for partial matches or synonyms
        classification_words = classification_lower.split()
        matches = sum(1 for word in classification_words if word in all_text)
        
        if matches > 0:
            return 0.3 + (matches / len(classification_words)) * 0.3
        
        return 0.1  # Default low confidence
    
    def _combine_confidence_scores(self, rule_confidence: float, 
                                 llm_confidence: float) -> float:
        """Combine rule-based and LLM confidence scores."""
        # Weighted average with slight preference for rule-based scoring
        rule_weight = 0.6
        llm_weight = 0.4
        
        combined = (rule_confidence * rule_weight) + (llm_confidence * llm_weight)
        
        # Boost if both agree (both high confidence)
        if rule_confidence > 0.6 and llm_confidence > 0.6:
            combined = min(1.0, combined + 0.1)
        
        # Reduce if they strongly disagree
        if abs(rule_confidence - llm_confidence) > 0.5:
            combined = combined * 0.8
        
        return max(0.0, min(1.0, combined))
    
    def get_confidence_summary(self, all_results: Dict[str, Dict]) -> Dict:
        """Generate a summary of confidence scores across all facets."""
        confidences = [result['confidence'] for result in all_results.values()]
        
        if not confidences:
            return {'overall_confidence': 0.0, 'high_confidence_count': 0}
        
        return {
            'overall_confidence': sum(confidences) / len(confidences),
            'high_confidence_count': sum(1 for c in confidences if c >= 0.7),
            'medium_confidence_count': sum(1 for c in confidences if 0.4 <= c < 0.7),
            'low_confidence_count': sum(1 for c in confidences if c < 0.4),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences)
        }