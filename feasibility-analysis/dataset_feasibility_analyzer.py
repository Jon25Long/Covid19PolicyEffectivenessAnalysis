"""
Dataset Feasibility Analyzer Framework - LEGACY VERSION
========================================================

IMPORTANT: This is the LEGACY version with HARDCODED column synonyms.
For new projects, use dynamic_feasibility_analyzer.py instead!

WHY THIS FILE EXISTS:
---------------------
1. EDUCATIONAL: Demonstrates the "before" in a before/after refactoring story
2. COMPARISON: Shows tradeoffs between manual curation vs algorithmic discovery
3. BACKWARD COMPATIBILITY: Existing code may depend on this implementation
4. DOMAIN EXPERTISE: Hardcoded synonyms can capture domain-specific knowledge

KEY LIMITATION:
---------------
This analyzer uses a HARDCODED COLUMN_SYNONYMS dictionary (see line ~130).
This means:
  ❌ Domain-specific (healthcare-focused)
  ❌ Requires manual updates when switching to finance/environmental/other data
  ❌ Misses synonyms not explicitly listed
  ❌ Maintenance burden grows with each new dataset

MODERN ALTERNATIVE:
-------------------
See: dynamic_feasibility_analyzer.py
  ✅ Uses SemanticColumnMatcher for runtime synonym discovery
  ✅ Domain-agnostic (works on ANY data)
  ✅ Zero hardcoded column names
  ✅ Automatically adapts to new datasets

WHEN TO USE THIS FILE:
----------------------
- Working with healthcare data where synonyms are well-known
- Need human curation of specific synonym relationships
- Integrating with existing code that expects this API
- Comparing hardcoded vs dynamic approaches

WHEN TO USE dynamic_feasibility_analyzer.py:
--------------------------------------------
- Starting a new project
- Working with non-healthcare data
- Want zero-maintenance solution
- Need to demonstrate domain independence for portfolio

Framework Overview:
-------------------
A generalizable framework for discovering research opportunities in multi-dataset
collections. Automatically identifies feasible research questions by analyzing
data completeness, alignment, and statistical viability.

This module inverts the traditional research workflow:
- Traditional: Researcher defines question → seeks data → discovers data is inadequate
- This tool: Data available → tool suggests feasible questions → researcher refines

Author: Derived from globalhealthdata project
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re


class FeasibilityLevel(Enum):
    """Classification of study feasibility based on data completeness."""
    HIGH = "HIGH"           # >80% complete - proceed with confidence
    MEDIUM = "MEDIUM"       # 60-80% complete - feasible with caveats
    LOW = "LOW"             # 40-60% complete - marginal, high risk
    INFEASIBLE = "INFEASIBLE"  # <40% complete - not recommended


class VariableRole(Enum):
    """Semantic role of a variable in potential analyses."""
    ENTITY = "entity"           # Country, Region, Hospital, Patient ID
    TEMPORAL = "temporal"       # Year, Quarter, Date, Month
    OUTCOME = "outcome"         # Health outcomes, mortality, life expectancy
    PREDICTOR = "predictor"     # Potential independent variables
    CONFOUNDER = "confounder"   # Variables that should be controlled
    DEMOGRAPHIC = "demographic" # Population characteristics
    RESOURCE = "resource"       # Healthcare resources, economic indicators
    UNKNOWN = "unknown"         # Needs classification


@dataclass
class ColumnMetadata:
    """Metadata about a discovered column."""
    name: str
    role: VariableRole
    data_type: str
    unique_count: int
    null_percent: float
    sample_values: List
    inferred_semantic_type: Optional[str] = None
    
    def __repr__(self):
        return f"{self.name} ({self.role.value}, {100-self.null_percent:.1f}% complete)"


@dataclass
class DatasetAlignment:
    """Analysis of how well two datasets align."""
    dataset1: str
    dataset2: str
    entity_overlap_count: int
    entity_overlap_percent: float
    temporal_overlap_count: int
    temporal_overlap_years: List
    common_entities: Set[str]
    completeness_score: float  # 0-100, weighted average of coverage
    
    def is_feasible(self, threshold: float = 60.0) -> bool:
        """Check if alignment meets minimum feasibility threshold."""
        return self.completeness_score >= threshold


@dataclass
class StudyOpportunity:
    """A potentially feasible research question discovered in the data."""
    title: str
    description: str
    feasibility: FeasibilityLevel
    datasets_required: List[str]
    outcome_variables: List[str]
    predictor_variables: List[str]
    confounders: List[str]
    temporal_range_recommended: Tuple[int, int]
    entity_count: int
    completeness_percent: float
    sample_size_adequate: bool
    statistical_power: Optional[float] = None
    caveats: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"[{self.feasibility.value}] {self.title} (n={self.entity_count}, {self.completeness_percent:.1f}% complete)"


class DatasetFeasibilityAnalyzer:
    """
    Main analyzer class for discovering feasible research opportunities.
    
    Key Design Decisions (based on public health best practices):
    
    1. Completeness Thresholds:
       - HIGH (>80%): Standard for publication-quality observational studies
       - MEDIUM (60-80%): Acceptable with sensitivity analyses and missingness discussion
       - LOW (40-60%): Exploratory only, high risk of bias
       - INFEASIBLE (<40%): Not recommended, insufficient statistical power
       
    2. Minimum Sample Sizes (epidemiological guidelines):
       - Cross-sectional: n≥30 for basic analyses, n≥50 preferred
       - Regression with controls: n≥10 per predictor variable (rule of thumb)
       - Time series: n≥20 time points for trend detection
       - Panel data: n×t ≥ 100 observations minimum
       
    3. Fuzzy Matching (for column name variations):
       - Threshold: 0.8 similarity (Levenshtein distance)
       - Case-insensitive, removes special characters
       - Common synonyms hard-coded (Country/Nation, GDP/GNI, etc.)
       
    4. Semantic Classification (domain knowledge):
       - Outcome variables: Keywords like "mortality", "expectancy", "rate", "incidence"
       - Predictors: "GDP", "doctors", "education", "expenditure"
       - Demographics: "population", "age", "gender", "urban"
       
    5. Statistical Power (Cohen's guidelines):
       - Small effect (r=0.1): Requires n≥783 for 80% power
       - Medium effect (r=0.3): Requires n≥84 for 80% power  
       - Large effect (r=0.5): Requires n≥28 for 80% power
       - Default assumption: Medium effect size for health correlations
    """
    
    # Public health domain knowledge
    OUTCOME_KEYWORDS = [
        'mortality', 'death', 'life_expectancy', 'lifespan', 'survival',
        'incidence', 'prevalence', 'disease', 'infection', 'illness',
        'health_outcome', 'morbidity', 'disability'
    ]
    
    PREDICTOR_KEYWORDS = [
        'gdp', 'gni', 'income', 'wealth', 'economic', 'expenditure',
        'doctors', 'physicians', 'nurses', 'hospital', 'clinic',
        'education', 'literacy', 'school', 'university',
        'sanitation', 'water', 'infrastructure'
    ]
    
    DEMOGRAPHIC_KEYWORDS = [
        'population', 'age', 'gender', 'sex', 'male', 'female',
        'urban', 'rural', 'density', 'race', 'ethnicity'
    ]
    
    ENTITY_KEYWORDS = [
        'country', 'nation', 'state', 'region', 'territory',
        'city', 'county', 'district', 'province',
        'id', 'code', 'identifier'
    ]
    
    TEMPORAL_KEYWORDS = [
        'year', 'date', 'time', 'period', 'quarter', 'month', 'day'
    ]
    
    # =========================================================================
    # HARDCODED SYNONYMS - THIS IS THE KEY LIMITATION!
    # =========================================================================
    # Column name synonyms for fuzzy matching
    #
    # PROBLEM: This dictionary requires MANUAL maintenance!
    #   - Must add new synonyms when encountering new datasets
    #   - Domain-specific (healthcare-focused)
    #   - Breaks when switching to finance, environmental, clinical data
    #   - Misses synonyms not explicitly listed here
    #
    # SOLUTION: Use dynamic_feasibility_analyzer.py instead!
    #   - Discovers synonyms automatically using SemanticColumnMatcher
    #   - Works on ANY domain without modification
    #   - Zero maintenance burden
    #
    # WHY IT'S STILL HERE:
    #   - Educational: Shows "before" refactoring
    #   - Backward compatibility: Existing code may depend on this
    #   - Domain expertise: Can capture human-curated relationships
    # =========================================================================
    COLUMN_SYNONYMS = {
        'country': ['nation', 'country_name', 'countryname'],
        'year': ['yr', 'year_val', 'time_year'],
        'gdp': ['gni', 'gdp_per_capita', 'gdp_pc', 'gross_domestic_product'],
        'population': ['pop', 'total_population', 'total_pop'],
        'life_expectancy': ['life_expect', 'lifespan', 'longevity']
    }
    
    def __init__(self, 
                 file_paths: List[str] = None,
                 dataframes: Dict[str, pd.DataFrame] = None,
                 min_completeness_threshold: float = 60.0,
                 min_sample_size: int = 30,
                 fuzzy_match_threshold: float = 0.8):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        file_paths : List[str], optional
            List of CSV file paths to analyze
        dataframes : Dict[str, pd.DataFrame], optional
            Dictionary of pre-loaded dataframes {name: df}
        min_completeness_threshold : float
            Minimum % completeness for feasible studies (default: 60%)
        min_sample_size : int
            Minimum entities required for analysis (default: 30, per epidemiological guidelines)
        fuzzy_match_threshold : float
            Similarity threshold for column name matching (default: 0.8)
        """
        self.min_completeness = min_completeness_threshold
        self.min_sample_size = min_sample_size
        self.fuzzy_threshold = fuzzy_match_threshold
        
        # Load datasets
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_metadata: Dict[str, Dict[str, ColumnMetadata]] = {}
        
        if dataframes:
            self.datasets = dataframes
        elif file_paths:
            self._load_datasets(file_paths)
        
        # Analysis results
        self.schema_analysis: Dict = {}
        self.alignment_matrix: Dict[Tuple[str, str], DatasetAlignment] = {}
        self.study_opportunities: List[StudyOpportunity] = []
    
    def _load_datasets(self, file_paths: List[str]):
        """Load CSV files into dataframes."""
        for path in file_paths:
            path_obj = Path(path)
            name = path_obj.stem  # filename without extension
            try:
                df = pd.read_csv(path)
                self.datasets[name] = df
                print(f"✓ Loaded {name}: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ Failed to load {path}: {e}")
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using Levenshtein distance.
        Returns value between 0 (completely different) and 1 (identical).
        """
        str1 = str1.lower().replace('_', '').replace(' ', '')
        str2 = str2.lower().replace('_', '').replace(' ', '')
        
        if str1 == str2:
            return 1.0
        
        # Simple Levenshtein distance calculation
        len1, len2 = len(str1), len(str2)
        if len1 > len2:
            str1, str2 = str2, str1
            len1, len2 = len2, len1
        
        distances = range(len1 + 1)
        for i2, char2 in enumerate(str2):
            new_distances = [i2 + 1]
            for i1, char1 in enumerate(str1):
                if char1 == char2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
            distances = new_distances
        
        max_len = max(len(str1), len(str2))
        return 1 - (distances[-1] / max_len) if max_len > 0 else 0.0
    
    def _check_synonym(self, col_name: str, target_name: str) -> bool:
        """
        Check if two column names are semantically equivalent.
        Uses synonym dictionary + fuzzy matching + semantic keywords.
        """
        col_clean = col_name.lower().replace('_', ' ').replace('-', ' ').strip()
        target_clean = target_name.lower().replace('_', ' ').replace('-', ' ').strip()
        
        # Direct match (normalized)
        if col_clean == target_clean:
            return True
        
        # Remove all whitespace/underscores for comparison
        col_normalized = col_clean.replace(' ', '')
        target_normalized = target_clean.replace(' ', '')
        
        if col_normalized == target_normalized:
            return True
        
        # Check if both belong to same semantic category using entity keywords
        col_is_entity = any(keyword in col_clean for keyword in self.ENTITY_KEYWORDS)
        target_is_entity = any(keyword in target_clean for keyword in self.ENTITY_KEYWORDS)
        
        if col_is_entity and target_is_entity:
            # Both are entity columns - they match!
            return True
        
        # Check synonym dictionary (bidirectional)
        for key, synonyms in self.COLUMN_SYNONYMS.items():
            # Check if both columns match the same synonym group
            col_matches = (col_normalized == key.replace('_', '') or 
                          any(col_normalized == syn.replace('_', '').replace(' ', '') for syn in synonyms))
            target_matches = (target_normalized == key.replace('_', '') or
                             any(target_normalized == syn.replace('_', '').replace(' ', '') for syn in synonyms))
            
            if col_matches and target_matches:
                return True
        
        # Fuzzy string matching as fallback
        similarity = self._levenshtein_similarity(col_clean, target_clean)
        if similarity >= self.fuzzy_match_threshold:
            return True
        
        return False
    
    def _infer_column_role(self, col_name: str, sample_values: List, data_type: str) -> VariableRole:
        """
        Infer semantic role of a column based on name and content.
        Uses public health domain knowledge.
        """
        col_lower = col_name.lower()
        col_normalized = col_lower.replace('_', ' ').replace('-', ' ')
        
        # Check entity keywords (must be whole words or at start/end to avoid false matches)
        entity_keywords_precise = ['country', 'nation', 'state', 'region', 'territory',
                                   'city', 'county', 'district', 'province']
        entity_keywords_suffix = ['code', 'identifier']
        
        # Whole word match for main entity keywords
        col_words = col_normalized.split()
        if any(keyword in col_words for keyword in entity_keywords_precise):
            return VariableRole.ENTITY
        
        # Suffix match for codes/IDs (but not if it's part of a longer word)
        if any(col_normalized.endswith(keyword) for keyword in entity_keywords_suffix):
            return VariableRole.ENTITY
        
        # "id" only if it's standalone or at start (country_id yes, incidence no)
        if col_normalized == 'id' or col_normalized.startswith('id ') or col_normalized.endswith(' id'):
            return VariableRole.ENTITY
        
        # Check temporal keywords
        if any(keyword in col_lower for keyword in self.TEMPORAL_KEYWORDS):
            return VariableRole.TEMPORAL
        
        # Check if it's numeric year values in column headers (wide format)
        if data_type in ['int64', 'float64'] and col_name.isdigit():
            year_val = int(col_name)
            if 1900 <= year_val <= 2100:
                return VariableRole.TEMPORAL
        
        # Check outcome keywords
        if any(keyword in col_lower for keyword in self.OUTCOME_KEYWORDS):
            return VariableRole.OUTCOME
        
        # Check predictor keywords
        if any(keyword in col_lower for keyword in self.PREDICTOR_KEYWORDS):
            return VariableRole.PREDICTOR
        
        # Check demographic keywords
        if any(keyword in col_lower for keyword in self.DEMOGRAPHIC_KEYWORDS):
            return VariableRole.DEMOGRAPHIC
        
        # Resource indicators (often confounders)
        if 'resource' in col_lower or 'expenditure' in col_lower or 'investment' in col_lower:
            return VariableRole.RESOURCE
        
        return VariableRole.UNKNOWN
    
    def analyze_schema(self) -> Dict:
        """
        Phase 1: Discover schema of all datasets.
        Returns metadata about columns and their inferred roles.
        """
        print("=" * 80)
        print("PHASE 1: SCHEMA DISCOVERY")
        print("=" * 80)
        
        for name, df in self.datasets.items():
            print(f"\nAnalyzing dataset: {name}")
            self.dataset_metadata[name] = {}
            
            for col in df.columns:
                # Gather column statistics
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                unique_count = df[col].nunique()
                data_type = str(df[col].dtype)
                
                # Get sample values (non-null)
                sample_vals = df[col].dropna().head(5).tolist()
                
                # Infer role
                role = self._infer_column_role(col, sample_vals, data_type)
                
                # Create metadata
                metadata = ColumnMetadata(
                    name=col,
                    role=role,
                    data_type=data_type,
                    unique_count=unique_count,
                    null_percent=null_pct,
                    sample_values=sample_vals
                )
                
                self.dataset_metadata[name][col] = metadata
                
                # Print summary
                print(f"  {col:40} | {role.value:12} | {100-null_pct:5.1f}% complete | {unique_count:5} unique")
        
        self.schema_analysis = {
            'datasets': list(self.datasets.keys()),
            'total_columns': sum(len(meta) for meta in self.dataset_metadata.values()),
            'columns_by_role': self._summarize_columns_by_role()
        }
        
        return self.schema_analysis
    
    def _summarize_columns_by_role(self) -> Dict[VariableRole, int]:
        """Count columns by their inferred role across all datasets."""
        role_counts = {role: 0 for role in VariableRole}
        for dataset_meta in self.dataset_metadata.values():
            for col_meta in dataset_meta.values():
                role_counts[col_meta.role] += 1
        return role_counts
    
    def find_entity_alignments(self) -> Dict[str, Dict]:
        """
        Find common entity columns across datasets (e.g., Country, Region).
        Returns mapping of standardized names to actual column names in each dataset.
        """
        print("\n" + "=" * 80)
        print("FINDING ENTITY COLUMN ALIGNMENTS")
        print("=" * 80)
        
        entity_alignments = {}
        
        # Get all entity columns
        entity_cols_by_dataset = {}
        for dataset_name, metadata in self.dataset_metadata.items():
            entity_cols = [col_meta for col_meta in metadata.values() 
                          if col_meta.role == VariableRole.ENTITY]
            entity_cols_by_dataset[dataset_name] = entity_cols
        
        # Find matches across datasets
        datasets = list(self.datasets.keys())
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                cols1 = entity_cols_by_dataset.get(dataset1, [])
                cols2 = entity_cols_by_dataset.get(dataset2, [])
                
                for col1 in cols1:
                    for col2 in cols2:
                        # Check if columns match (synonym or fuzzy match)
                        if self._check_synonym(col1.name, col2.name):
                            match_key = f"{col1.name}|{col2.name}"
                            if match_key not in entity_alignments:
                                entity_alignments[match_key] = {
                                    'standard_name': col1.name,
                                    'datasets': {}
                                }
                            entity_alignments[match_key]['datasets'][dataset1] = col1.name
                            entity_alignments[match_key]['datasets'][dataset2] = col2.name
                            print(f"✓ Entity match: {dataset1}.{col1.name} ↔ {dataset2}.{col2.name}")
        
        return entity_alignments
    
    def find_temporal_alignments(self) -> Dict[str, Dict]:
        """
        Find common temporal columns across datasets (e.g., Year).
        Handles both long format (Year column) and wide format (year as column headers).
        """
        print("\n" + "=" * 80)
        print("FINDING TEMPORAL COLUMN ALIGNMENTS")
        print("=" * 80)
        
        temporal_alignments = {}
        
        for dataset_name, metadata in self.dataset_metadata.items():
            df = self.datasets[dataset_name]
            
            # First check for wide format (year columns like '2000', '2001', etc.)
            year_cols = [col for col in df.columns if str(col).isdigit()]
            if len(year_cols) > 5:  # If we have many numeric columns, likely wide format
                years = sorted([int(col) for col in year_cols])
                # Validate these are actually years (1900-2100 range)
                valid_years = [y for y in years if 1900 <= y <= 2100]
                if len(valid_years) >= 5:
                    print(f"  {dataset_name}: Wide format with {len(valid_years)} year columns ({min(valid_years)}-{max(valid_years)})")
                    temporal_alignments[dataset_name] = {
                        'format': 'wide',
                        'column': None,
                        'years': valid_years
                    }
                    continue  # Skip long-format check for this dataset
            
            # Otherwise check for long format (Year column with year values)
            temporal_cols = [col_meta for col_meta in metadata.values() 
                           if col_meta.role == VariableRole.TEMPORAL and 
                           not str(col_meta.name).isdigit()]  # Exclude numeric column names
            
            if temporal_cols:
                for col_meta in temporal_cols:
                    if col_meta.data_type in ['int64', 'float64']:
                        # Long format: Year column with year values
                        years = sorted(df[col_meta.name].dropna().unique())
                        # Validate these are actually years
                        valid_years = [int(y) for y in years if 1900 <= y <= 2100]
                        if len(valid_years) > 0:
                            print(f"  {dataset_name}.{col_meta.name}: {min(valid_years)}-{max(valid_years)} ({len(valid_years)} years)")
                            temporal_alignments[dataset_name] = {
                                'format': 'long',
                                'column': col_meta.name,
                                'years': valid_years
                            }
        
        return temporal_alignments
    
    def compute_alignment_matrix(self) -> Dict[Tuple[str, str], DatasetAlignment]:
        """
        Phase 2: Compute how well each pair of datasets aligns.
        Returns alignment metrics for all dataset pairs.
        """
        print("\n" + "=" * 80)
        print("PHASE 2: DATASET ALIGNMENT ANALYSIS")
        print("=" * 80)
        
        # First, find entity and temporal alignments
        entity_aligns = self.find_entity_alignments()
        temporal_aligns = self.find_temporal_alignments()
        
        # Now compute pairwise alignments
        datasets = list(self.datasets.keys())
        
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                alignment = self._compute_pairwise_alignment(
                    dataset1, dataset2, entity_aligns, temporal_aligns
                )
                self.alignment_matrix[(dataset1, dataset2)] = alignment
                
                # Print summary
                print(f"\n{dataset1} ↔ {dataset2}:")
                print(f"  Entity overlap: {alignment.entity_overlap_count} ({alignment.entity_overlap_percent:.1f}%)")
                print(f"  Temporal overlap: {len(alignment.temporal_overlap_years)} years")
                print(f"  Completeness score: {alignment.completeness_score:.1f}%")
                print(f"  Feasible: {'✓ YES' if alignment.is_feasible(self.min_completeness) else '✗ NO'}")
        
        return self.alignment_matrix
    
    def _compute_pairwise_alignment(self, 
                                   dataset1: str, 
                                   dataset2: str,
                                   entity_aligns: Dict,
                                   temporal_aligns: Dict) -> DatasetAlignment:
        """Compute detailed alignment metrics for a pair of datasets."""
        
        df1 = self.datasets[dataset1]
        df2 = self.datasets[dataset2]
        
        # Find common entity column
        common_entity_col = None
        for align_key, align_info in entity_aligns.items():
            if dataset1 in align_info['datasets'] and dataset2 in align_info['datasets']:
                entity_col1 = align_info['datasets'][dataset1]
                entity_col2 = align_info['datasets'][dataset2]
                common_entity_col = (entity_col1, entity_col2)
                break
        
        # Compute entity overlap
        if common_entity_col:
            entities1 = set(df1[common_entity_col[0]].dropna().unique())
            entities2 = set(df2[common_entity_col[1]].dropna().unique())
            common_entities = entities1 & entities2
            entity_overlap_pct = (len(common_entities) / max(len(entities1), len(entities2))) * 100
        else:
            common_entities = set()
            entity_overlap_pct = 0.0
        
        # Compute temporal overlap
        temporal1 = temporal_aligns.get(dataset1, {})
        temporal2 = temporal_aligns.get(dataset2, {})
        
        if temporal1 and temporal2:
            years1 = set(temporal1['years'])
            years2 = set(temporal2['years'])
            common_years = sorted(years1 & years2)
        else:
            common_years = []
        
        # Completeness score (simplified: average of entity and temporal overlap)
        completeness = (entity_overlap_pct + (len(common_years) / 20 * 100)) / 2 if common_years else entity_overlap_pct / 2
        
        return DatasetAlignment(
            dataset1=dataset1,
            dataset2=dataset2,
            entity_overlap_count=len(common_entities),
            entity_overlap_percent=entity_overlap_pct,
            temporal_overlap_count=len(common_years),
            temporal_overlap_years=common_years,
            common_entities=common_entities,
            completeness_score=completeness
        )
    
    def generate_feasibility_report(self) -> str:
        """
        Generate a comprehensive feasibility report.
        Summarizes what research questions are feasible given the available data.
        """
        if not self.schema_analysis:
            self.analyze_schema()
        
        if not self.alignment_matrix:
            self.compute_alignment_matrix()
        
        report = []
        report.append("=" * 80)
        report.append("DATASET FEASIBILITY ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary
        report.append(f"\nDATASETS ANALYZED: {len(self.datasets)}")
        for name, df in self.datasets.items():
            report.append(f"  • {name}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Schema summary
        report.append(f"\nCOLUMN CLASSIFICATION:")
        role_counts = self.schema_analysis['columns_by_role']
        for role, count in role_counts.items():
            if count > 0:
                report.append(f"  • {role.value.capitalize()}: {count} columns")
        
        # Alignment summary
        report.append(f"\nDATASET ALIGNMENT ANALYSIS:")
        high_feasibility = [a for a in self.alignment_matrix.values() if a.completeness_score >= 80]
        medium_feasibility = [a for a in self.alignment_matrix.values() if 60 <= a.completeness_score < 80]
        low_feasibility = [a for a in self.alignment_matrix.values() if 40 <= a.completeness_score < 60]
        infeasible = [a for a in self.alignment_matrix.values() if a.completeness_score < 40]
        
        report.append(f"\n  HIGH FEASIBILITY (>80% complete): {len(high_feasibility)} pairs")
        for align in high_feasibility:
            report.append(f"    ✓ {align.dataset1} ↔ {align.dataset2}: "
                         f"{align.entity_overlap_count} entities, "
                         f"{len(align.temporal_overlap_years)} years, "
                         f"{align.completeness_score:.1f}% complete")
        
        report.append(f"\n  MEDIUM FEASIBILITY (60-80% complete): {len(medium_feasibility)} pairs")
        for align in medium_feasibility:
            report.append(f"    ⚠ {align.dataset1} ↔ {align.dataset2}: "
                         f"{align.entity_overlap_count} entities, "
                         f"{len(align.temporal_overlap_years)} years, "
                         f"{align.completeness_score:.1f}% complete")
        
        if low_feasibility:
            report.append(f"\n  LOW FEASIBILITY (40-60% complete): {len(low_feasibility)} pairs")
            for align in low_feasibility:
                report.append(f"    ⚠ {align.dataset1} ↔ {align.dataset2}: {align.completeness_score:.1f}% complete")
        
        if infeasible:
            report.append(f"\n  INFEASIBLE (<40% complete): {len(infeasible)} pairs")
            for align in infeasible:
                report.append(f"    ✗ {align.dataset1} ↔ {align.dataset2}: {align.completeness_score:.1f}% complete")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Convenience function for quick analysis
def quick_analyze(file_paths: List[str] = None, dataframes: Dict[str, pd.DataFrame] = None):
    """
    Quick feasibility analysis of CSV files or dataframes.
    
    Usage:
        quick_analyze(['file1.csv', 'file2.csv', 'file3.csv'])
        
    Or:
        quick_analyze(dataframes={'health': df1, 'doctors': df2})
    """
    analyzer = DatasetFeasibilityAnalyzer(file_paths=file_paths, dataframes=dataframes)
    analyzer.analyze_schema()
    analyzer.compute_alignment_matrix()
    print("\n" + analyzer.generate_feasibility_report())
    return analyzer
