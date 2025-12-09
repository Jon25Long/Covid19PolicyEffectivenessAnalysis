"""
Model Profiles for Domain-Specific Column Matching
===================================================

Predefined configurations for different neural embedding models
optimized for specific public health domains and analysis types.

Usage:
------
    from model_profiles import MODEL_PROFILES, recommend_model
    
    # Get recommendation
    profile = recommend_model(
        domains=['clinical', 'epidemiology'],
        analysis_type='retrospective'
    )
    
    # Use in analyzer
    analyzer = DynamicFeasibilityAnalyzer(
        data_dir='Data/',
        model_profile=profile
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ModelProfile:
    """Configuration profile for neural model selection."""
    name: str
    model_id: str
    size_mb: int
    speed: str  # 'fast', 'medium', 'slow'
    domains: List[str]
    use_cases: List[str]
    caveats: List[str]
    requires_preprocessing: bool = False
    recommended_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.recommended_thresholds:
            self.recommended_thresholds = {
                'min_structural_similarity': 0.3,
                'min_token_overlap': 0.3,
                'min_value_overlap': 0.2
            }


# Predefined model profiles
MODEL_PROFILES = {
    'general_default': ModelProfile(
        name='General Purpose (Default)',
        model_id='all-MiniLM-L6-v2',
        size_mb=90,
        speed='fast',
        domains=['any'],
        use_cases=[
            'First-time exploration',
            'Mixed-domain data',
            'Portfolio demonstrations',
            'Retrospective analysis',
            'Quick iterations'
        ],
        caveats=[
            'May miss highly specialized acronyms',
            'Less effective on rare medical terms',
            '~98% relationship discovery (vs 100% for premium)'
        ],
        recommended_thresholds={
            'min_structural_similarity': 0.3,
            'min_token_overlap': 0.3,
            'min_value_overlap': 0.2
        }
    ),
    
    'general_premium': ModelProfile(
        name='General Purpose (Premium)',
        model_id='all-mpnet-base-v2',
        size_mb=420,
        speed='medium',
        domains=['any'],
        use_cases=[
            'Maximum accuracy required',
            'Social determinants of health (SDOH)',
            'Complex multi-source data',
            'Final validation passes',
            'Portfolio showcase (best practices)'
        ],
        caveats=[
            'Larger download (420MB vs 90MB)',
            'Slower inference (~2x vs default)',
            'Only marginal improvement for single-domain data'
        ],
        recommended_thresholds={
            'min_structural_similarity': 0.3,
            'min_token_overlap': 0.3,
            'min_value_overlap': 0.2
        }
    ),
    
    'public_health_clinical': ModelProfile(
        name='Public Health - Clinical/Biomedical',
        model_id='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        size_mb=420,
        speed='medium',
        domains=['public_health', 'clinical', 'epidemiology', 'biomedical'],
        use_cases=[
            'Disease surveillance',
            'Immunization records',
            'Clinical trials',
            'Mortality/morbidity data',
            'Healthcare quality metrics',
            'Epidemiological studies'
        ],
        caveats=[
            'May over-match medical synonyms (tighter clustering)',
            'Adjust thresholds higher (0.4+ for structural similarity)',
            'Limited on non-medical columns (demographics, economics)',
            'Consider hybrid approach for mixed-domain data'
        ],
        recommended_thresholds={
            'min_structural_similarity': 0.4,  # Higher due to tight medical term clustering
            'min_token_overlap': 0.3,
            'min_value_overlap': 0.2
        }
    ),
    
    'public_health_environmental': ModelProfile(
        name='Public Health - Environmental',
        model_id='climatebert/distilroberta-base-climate-f',
        size_mb=310,
        speed='fast',
        domains=['environmental_health', 'climate', 'ecology', 'air_quality'],
        use_cases=[
            'Air quality monitoring',
            'Climate-health impacts',
            'Environmental justice',
            'Pollution exposure studies',
            'Geographic health analysis'
        ],
        caveats=[
            'Specialized for climate/environmental science',
            'May miss clinical health terminology',
            'Consider hybrid with general model for mixed data',
            'Strong on emissions, pollution, temperature terms'
        ],
        recommended_thresholds={
            'min_structural_similarity': 0.3,
            'min_token_overlap': 0.3,
            'min_value_overlap': 0.2
        }
    ),
    
    'health_economics': ModelProfile(
        name='Health Economics',
        model_id='ProsusAI/finbert',
        size_mb=420,
        speed='medium',
        domains=['health_economics', 'finance', 'socioeconomic'],
        use_cases=[
            'GDP vs health outcomes',
            'Healthcare expenditure analysis',
            'Economic determinants of health',
            'Insurance/access studies',
            'Cost-effectiveness analysis'
        ],
        caveats=[
            'Financial model, not health-specific',
            'Blend with general model for clinical terms',
            'Strong on economic indicators, weak on medical terminology',
            'Best for economics-heavy datasets only'
        ],
        recommended_thresholds={
            'min_structural_similarity': 0.3,
            'min_token_overlap': 0.3,
            'min_value_overlap': 0.2
        }
    ),
    
    'genetics_genomics': ModelProfile(
        name='Genetics & Genomics',
        model_id='dmis-lab/biobert-base-cased-v1.2',
        size_mb=420,
        speed='medium',
        domains=['genomics', 'genetics', 'molecular_epidemiology'],
        use_cases=[
            'Genetic markers',
            'Pharmacogenomics',
            'Gene-environment interactions',
            'Precision medicine data',
            'SNP/gene expression studies'
        ],
        caveats=[
            'Case-sensitive (critical for gene names: TP53 vs tp53)',
            'Research-oriented vocabulary',
            'Limited on operational public health terms',
            'Requires preserving case in column names'
        ],
        requires_preprocessing=True,
        recommended_thresholds={
            'min_structural_similarity': 0.3,
            'min_token_overlap': 0.3,
            'min_value_overlap': 0.2
        }
    )
}


def recommend_model(
    data_description: str = None,
    analysis_type: str = None,
    domains: List[str] = None,
    priority: str = 'accuracy'
) -> ModelProfile:
    """
    Recommend neural model based on use case.
    
    Parameters
    ----------
    data_description : str, optional
        Description of datasets (e.g., "WHO immunization + GDP data")
    analysis_type : str, optional
        Type of analysis: 'retrospective', 'predictive', 'exploratory'
    domains : List[str], optional
        Domains present in data: ['clinical', 'economic', 'environmental', etc.]
    priority : str, default='accuracy'
        Optimization priority: 'accuracy' or 'speed'
    
    Returns
    -------
    ModelProfile
        Recommended model configuration
    
    Examples
    --------
    >>> profile = recommend_model(
    ...     data_description="Immunization rates and mortality data",
    ...     analysis_type="retrospective",
    ...     domains=['clinical', 'epidemiology']
    ... )
    >>> print(profile.name)
    'Public Health - Clinical/Biomedical'
    
    >>> profile = recommend_model(
    ...     data_description="GDP, life expectancy, urban population",
    ...     domains=['economic', 'demographic']
    ... )
    >>> print(profile.name)
    'General Purpose (Premium)'
    """
    if domains is None:
        domains = []
    
    # Convert to lowercase for matching
    domain_set = set(d.lower() for d in domains)
    
    # Count domains to detect mixed vs single-domain
    domain_count = len(domain_set)
    
    # Domain-based recommendation
    
    # Clinical/Medical (single domain or dominant)
    clinical_domains = {'clinical', 'medical', 'epidemiology', 'disease', 'immunization', 'mortality'}
    if domain_set & clinical_domains:
        if domain_count == 1 or (len(domain_set & clinical_domains) / domain_count) > 0.6:
            if priority == 'accuracy':
                return MODEL_PROFILES['public_health_clinical']
    
    # Environmental (single domain or dominant)
    env_domains = {'environmental', 'climate', 'pollution', 'air_quality', 'emissions'}
    if domain_set & env_domains:
        if domain_count == 1 or (len(domain_set & env_domains) / domain_count) > 0.6:
            return MODEL_PROFILES['public_health_environmental']
    
    # Economics (single domain only - otherwise use general)
    econ_domains = {'economic', 'finance', 'gdp', 'income', 'expenditure'}
    if domain_set & econ_domains:
        if domain_count == 1:
            return MODEL_PROFILES['health_economics']
        # Mixed with other domains → general model better
    
    # Genetics (single domain or dominant)
    genetics_domains = {'genetics', 'genomics', 'molecular', 'gene', 'snp'}
    if domain_set & genetics_domains:
        if domain_count == 1 or (len(domain_set & genetics_domains) / domain_count) > 0.6:
            return MODEL_PROFILES['genetics_genomics']
    
    # Mixed domains or no clear specialization
    if domain_count > 1:
        # Multiple domains → general model handles all
        return MODEL_PROFILES['general_premium'] if priority == 'accuracy' else MODEL_PROFILES['general_default']
    
    # Analysis type fallback
    if analysis_type:
        if analysis_type.lower() in ['retrospective', 'historical']:
            return MODEL_PROFILES['general_premium'] if priority == 'accuracy' else MODEL_PROFILES['general_default']
        
        if analysis_type.lower() in ['predictive', 'forecasting', 'ml']:
            # Predictive needs maximum accuracy
            return MODEL_PROFILES['general_premium']
        
        if analysis_type.lower() in ['exploratory', 'discovery']:
            # Exploration prioritizes finding ALL relationships
            return MODEL_PROFILES['general_premium']
    
    # Default: general model with priority consideration
    return MODEL_PROFILES['general_premium'] if priority == 'accuracy' else MODEL_PROFILES['general_default']


def print_recommendation(profile: ModelProfile, rationale: str = None):
    """
    Pretty-print model recommendation with rationale.
    
    Parameters
    ----------
    profile : ModelProfile
        Model profile to display
    rationale : str, optional
        Custom rationale for this recommendation
    """
    print("=" * 80)
    print(f"RECOMMENDED MODEL: {profile.name}")
    print("=" * 80)
    print(f"\n📦 Model ID: {profile.model_id}")
    print(f"💾 Size: {profile.size_mb} MB")
    print(f"⚡ Speed: {profile.speed}")
    print(f"\n🎯 Domains: {', '.join(profile.domains)}")
    
    print(f"\n✅ Use Cases:")
    for uc in profile.use_cases:
        print(f"   • {uc}")
    
    if profile.caveats:
        print(f"\n⚠️  Caveats:")
        for caveat in profile.caveats:
            print(f"   • {caveat}")
    
    print(f"\n⚙️  Recommended Thresholds:")
    for param, value in profile.recommended_thresholds.items():
        print(f"   {param}: {value}")
    
    if profile.requires_preprocessing:
        print(f"\n🔧 Preprocessing Required: Yes")
        print(f"   (See model documentation for details)")
    
    if rationale:
        print(f"\n💡 Rationale:")
        print(f"{rationale}")
    
    print("=" * 80)


def detect_domains_from_keywords(column_names: List[str]) -> List[str]:
    """
    Automatically detect domains from column names.
    
    Parameters
    ----------
    column_names : List[str]
        List of column names from datasets
    
    Returns
    -------
    List[str]
        Detected domain names
    
    Examples
    --------
    >>> columns = ['DPT Immunization', 'GDP', 'Life expectancy', 'CO2 emissions']
    >>> detect_domains_from_keywords(columns)
    ['clinical', 'economic', 'environmental']
    """
    # Domain keyword mappings
    domain_keywords = {
        'clinical': [
            'immunization', 'vaccination', 'disease', 'mortality', 'morbidity',
            'diagnosis', 'treatment', 'patient', 'clinical', 'medical', 'health',
            'doctor', 'hospital', 'death', 'life expectancy', 'survival'
        ],
        'economic': [
            'gdp', 'income', 'expenditure', 'cost', 'price', 'finance', 'budget',
            'economic', 'poverty', 'employment', 'wage', 'salary', 'revenue'
        ],
        'environmental': [
            'air quality', 'pollution', 'emissions', 'co2', 'temperature', 'climate',
            'environmental', 'ecology', 'exposure', 'particulate', 'ozone'
        ],
        'demographic': [
            'population', 'age', 'gender', 'race', 'ethnicity', 'urban', 'rural',
            'density', 'migration', 'demographic', 'census'
        ],
        'genetics': [
            'gene', 'snp', 'allele', 'chromosome', 'dna', 'rna', 'genomic',
            'genetic', 'mutation', 'variant', 'expression'
        ]
    }
    
    detected_domains = set()
    
    # Convert column names to lowercase for matching
    columns_lower = [col.lower() for col in column_names]
    all_columns_text = ' '.join(columns_lower)
    
    # Check each domain's keywords
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in all_columns_text:
                detected_domains.add(domain)
                break  # One match per domain is enough
    
    return sorted(list(detected_domains))


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("MODEL PROFILE EXAMPLES")
    print("=" * 80)
    print()
    
    # Example 1: Current portfolio data
    print("EXAMPLE 1: Current Portfolio Data")
    print("-" * 80)
    print("Data: WHO immunization rates, World Bank mortality, GDP indicators")
    print()
    
    # Detect domains from typical columns
    sample_columns = [
        'Country', 'Year', 'DPT Immunization', 'HepB3 Immunization',
        'Measles Immunization', 'Life expectancy', 'Mortality rate',
        'GDP', 'GDP per capita', 'Urban population'
    ]
    
    detected = detect_domains_from_keywords(sample_columns)
    print(f"Detected domains: {detected}")
    print()
    
    profile = recommend_model(
        data_description="Immunization + mortality + GDP + demographics",
        analysis_type="retrospective",
        domains=detected,
        priority='accuracy'
    )
    
    rationale = """
This dataset combines multiple domains:
- Clinical/epidemiology (immunization, mortality) - ~40%
- Economic (GDP indicators) - ~20%
- Demographic (population, urban/rural) - ~40%

For MIXED-DOMAIN retrospective analysis:
→ General Premium model is optimal
→ BiomedNLP-PubMedBERT would excel on clinical terms but underperform on economic/demographic
→ all-mpnet-base-v2 handles all domains well with best overall accuracy
"""
    
    print_recommendation(profile, rationale)
    
    print("\n\n")
    
    # Example 2: Specialized clinical data
    print("EXAMPLE 2: Disease Surveillance Data")
    print("-" * 80)
    print("Data: CDC disease reports, immunization records, clinical outcomes")
    print()
    
    clinical_columns = [
        'Disease', 'Case Count', 'Vaccination Status', 'Clinical Outcome',
        'Diagnosis Date', 'Treatment Protocol', 'Mortality'
    ]
    
    detected_clinical = detect_domains_from_keywords(clinical_columns)
    print(f"Detected domains: {detected_clinical}")
    print()
    
    profile_clinical = recommend_model(
        data_description="Disease surveillance and clinical outcomes",
        analysis_type="predictive",
        domains=detected_clinical,
        priority='accuracy'
    )
    
    rationale_clinical = """
This dataset is SINGLE-DOMAIN clinical/medical:
- Disease surveillance - 100% medical terminology
- Predictive analysis (outbreak forecasting)
- High specificity needed for medical terms

For SINGLE-DOMAIN clinical analysis:
→ BiomedNLP-PubMedBERT is optimal
→ Trained on 14M PubMed abstracts + 3M full-text articles
→ Understands medical abbreviations, disease names, clinical procedures
→ Superior for medical-specific synonym detection
"""
    
    print_recommendation(profile_clinical, rationale_clinical)
    
    print("\n\n")
    
    # Example 3: Environmental health
    print("EXAMPLE 3: Environmental Health Analysis")
    print("-" * 80)
    print("Data: Air quality monitoring, climate data, health outcomes")
    print()
    
    env_columns = [
        'PM2.5', 'Ozone', 'CO2 Emissions', 'Temperature', 'Air Quality Index',
        'Respiratory Illness', 'Asthma Rates'
        ]
    
    detected_env = detect_domains_from_keywords(env_columns)
    print(f"Detected domains: {detected_env}")
    print()
    
    profile_env = recommend_model(
        data_description="Air quality and respiratory health",
        analysis_type="exploratory",
        domains=detected_env,
        priority='accuracy'
    )
    
    rationale_env = """
This dataset combines environmental + clinical:
- Environmental (air quality, climate) - ~60%
- Clinical (respiratory illness) - ~40%

For ENVIRONMENTAL-DOMINANT analysis:
→ ClimateBERT would be ideal for pure environmental data
→ But mixed with health outcomes suggests general model
→ all-mpnet-base-v2 handles both environmental and health terms

Alternative: Hybrid approach (ClimateBERT + general model)
"""
    
    print_recommendation(profile_env, rationale_env)
