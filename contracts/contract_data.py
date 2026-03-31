"""
Synthetic Contract Dataset with Golden Clause Annotations
Contracts: NDA, SaaS Agreement, Employment Agreement
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class RiskLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    NONE   = "none"

class RiskType(str, Enum):
    LIABILITY_CAP        = "liability_cap"
    IP_OWNERSHIP         = "ip_ownership"
    TERMINATION          = "termination"
    INDEMNIFICATION      = "indemnification"
    GOVERNING_LAW        = "governing_law"
    DATA_PRIVACY         = "data_privacy"
    NON_COMPETE          = "non_compete"
    AMBIGUOUS_LANGUAGE   = "ambiguous_language"
    MISSING_PROTECTION   = "missing_protection"
    UNILATERAL_CHANGE    = "unilateral_change"
    AUTO_RENEWAL         = "auto_renewal"
    PAYMENT_TERMS        = "payment_terms"
    CONFIDENTIALITY      = "confidentiality"
    DISPUTE_RESOLUTION   = "dispute_resolution"
    CLEAN                = "clean"

@dataclass
class Clause:
    clause_id: str
    clause_type: str
    text: str
    risk_level: RiskLevel
    risk_type: RiskType
    is_missing_protection: bool
    annotation: str
    is_buried: bool = False

@dataclass
class Contract:
    contract_id: str
    contract_type: str
    title: str
    parties: dict
    clauses: list

# CONTRACT 1 — MUTUAL NDA (Easy Task)
NDA_CONTRACT = Contract(
    contract_id="C-001",
    contract_type="NDA",
    title="Mutual Non-Disclosure Agreement",
    parties={"disclosing": "Acme Corp", "receiving": "Beta Ventures LLC"},
    clauses=[
        Clause(
            clause_id="C001-01",
            clause_type="Definition of Confidential Information",
            text='"Confidential Information" means any data or information that is proprietary to the Disclosing Party and not generally known to the public, whether in tangible or intangible form, whenever and however disclosed.',
            risk_level=RiskLevel.LOW,
            risk_type=RiskType.CONFIDENTIALITY,
            is_missing_protection=False,
            annotation="Standard definition. Acceptable, though 'whenever and however disclosed' is broad — could capture oral disclosures without a written follow-up requirement.",
        ),
        Clause(
            clause_id="C001-02",
            clause_type="Obligations of Receiving Party",
            text="The Receiving Party agrees to hold Confidential Information in strict confidence and to take reasonable precautions to protect such information. The Receiving Party may disclose Confidential Information to its employees on a need-to-know basis.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.CONFIDENTIALITY,
            is_missing_protection=True,
            annotation="'Reasonable precautions' is vague — no minimum standard defined. Missing: requirement that employees be bound by equivalent confidentiality obligations.",
        ),
        Clause(
            clause_id="C001-03",
            clause_type="Term",
            text="This Agreement shall remain in effect for two (2) years from the Effective Date. Confidentiality obligations shall survive termination of this Agreement indefinitely.",
            risk_level=RiskLevel.LOW,
            risk_type=RiskType.CLEAN,
            is_missing_protection=False,
            annotation="Standard NDA term with indefinite confidentiality survival. Clean clause.",
        ),
        Clause(
            clause_id="C001-04",
            clause_type="Return of Information",
            text="Upon request by the Disclosing Party, the Receiving Party shall promptly return all Confidential Information. The Receiving Party may retain one archival copy for legal compliance purposes without restriction on use.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.CONFIDENTIALITY,
            is_missing_protection=False,
            annotation="CRITICAL: 'Without restriction on use' for the archival copy is extremely dangerous. It effectively creates an unrestricted license to use confidential information indefinitely.",
        ),
        Clause(
            clause_id="C001-05",
            clause_type="Governing Law",
            text="This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law provisions. Any disputes shall be resolved exclusively in the courts of New Castle County, Delaware.",
            risk_level=RiskLevel.LOW,
            risk_type=RiskType.GOVERNING_LAW,
            is_missing_protection=False,
            annotation="Standard Delaware governing law. Acceptable for most US entities.",
        ),
        Clause(
            clause_id="C001-06",
            clause_type="Remedies",
            text="Each party acknowledges that any breach of this Agreement may cause irreparable harm for which monetary damages would be an inadequate remedy. Accordingly, each party shall be entitled to seek equitable relief without the requirement of posting a bond or other security.",
            risk_level=RiskLevel.NONE,
            risk_type=RiskType.CLEAN,
            is_missing_protection=False,
            annotation="Standard injunctive relief clause. Clean.",
        ),
        Clause(
            clause_id="C001-07",
            clause_type="Liability Cap",
            text="In no event shall either party's total liability under this Agreement exceed the greater of (a) amounts paid under this Agreement in the prior twelve months or (b) one thousand US dollars ($1,000).",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.LIABILITY_CAP,
            is_missing_protection=False,
            annotation="HIGH RISK: $1,000 liability cap is inadequate for an NDA involving valuable trade secrets. Should be negotiated to reflect actual value of confidential information at risk.",
        ),
        Clause(
            clause_id="C001-08",
            clause_type="Dispute Resolution",
            text="All disputes arising under this Agreement shall be resolved by binding arbitration conducted by a single arbitrator selected by the Company at its sole discretion.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.DISPUTE_RESOLUTION,
            is_missing_protection=False,
            annotation="HIGH RISK: Arbitrator selected solely by the Company is not neutral. Should require mutual agreement or appointment by a neutral body (AAA/JAMS).",
        ),
    ]
)

# CONTRACT 2 — SAAS AGREEMENT (Medium Task)
SAAS_CONTRACT = Contract(
    contract_id="C-002",
    contract_type="SaaS",
    title="Software as a Service Subscription Agreement",
    parties={"vendor": "CloudSoft Inc", "customer": "Meridian Analytics Ltd"},
    clauses=[
        Clause(
            clause_id="C002-01",
            clause_type="License Grant",
            text="Subject to the terms herein, Vendor grants Customer a non-exclusive, non-transferable, limited license to access and use the Service solely for Customer's internal business purposes during the Subscription Term.",
            risk_level=RiskLevel.NONE,
            risk_type=RiskType.CLEAN,
            is_missing_protection=False,
            annotation="Standard SaaS license grant. Clean.",
        ),
        Clause(
            clause_id="C002-02",
            clause_type="Service Level Agreement",
            text="Vendor will use commercially reasonable efforts to make the Service available 99.5% of the time, excluding scheduled maintenance. Customer's sole remedy for downtime shall be service credits not to exceed 10% of monthly fees.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.LIABILITY_CAP,
            is_missing_protection=True,
            annotation="'Commercially reasonable efforts' weakens the 99.5% commitment. 10% credit cap is below industry standard (typically 25-30%). Missing: definition of scheduled maintenance windows and advance notice requirement.",
        ),
        Clause(
            clause_id="C002-03",
            clause_type="Data Ownership",
            text="Customer retains all rights to Customer Data. Vendor may use Customer Data in anonymized and aggregated form for product improvement and benchmarking purposes.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.DATA_PRIVACY,
            is_missing_protection=False,
            annotation="'Anonymized and aggregated' is not defined. If re-identification is possible, this creates a GDPR/CCPA risk. Should specify anonymization standard.",
        ),
        Clause(
            clause_id="C002-04",
            clause_type="Unilateral Modification",
            text="Vendor reserves the right to modify the Service, including removing features or changing pricing, at any time with thirty (30) days written notice to Customer. Continued use of the Service constitutes acceptance of such modifications.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.UNILATERAL_CHANGE,
            is_missing_protection=False,
            annotation="HIGH RISK: Vendor can remove features or raise prices with only 30-day notice. Missing: right to exit without penalty if modifications are materially adverse.",
        ),
        Clause(
            clause_id="C002-05",
            clause_type="Auto-Renewal",
            text="This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.AUTO_RENEWAL,
            is_missing_protection=False,
            annotation="90-day non-renewal notice is longer than typical (30-60 days). Customer may miss the window.",
        ),
        Clause(
            clause_id="C002-06",
            clause_type="Intellectual Property",
            text="Any feedback, suggestions, or ideas provided by Customer regarding the Service shall be deemed the exclusive property of Vendor and Customer hereby irrevocably assigns all rights in such feedback to Vendor without compensation.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.IP_OWNERSHIP,
            is_missing_protection=False,
            annotation="IP assignment of feedback is standard but 'irrevocably' with 'no compensation' is aggressive. Customer product roadmap suggestions become vendor property.",
        ),
        Clause(
            clause_id="C002-07",
            clause_type="Indemnification",
            text="Customer shall indemnify, defend, and hold harmless Vendor from any claims arising out of Customer's use of the Service, including claims by Customer's end users, without limitation.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.INDEMNIFICATION,
            is_missing_protection=True,
            annotation="HIGH RISK: 'Without limitation' on indemnification scope is dangerous. Missing: reciprocal indemnification from Vendor for IP infringement claims. This is a one-sided indemnification.",
        ),
        Clause(
            clause_id="C002-08",
            clause_type="Limitation of Liability",
            text="In no event shall Vendor be liable for any indirect, incidental, special, or consequential damages. Vendor's total liability shall not exceed fees paid in the three (3) months preceding the claim.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.LIABILITY_CAP,
            is_missing_protection=False,
            annotation="HIGH RISK: 3-month fee cap is very low for enterprise SaaS. Excludes consequential damages but Customer's indemnification (C002-07) has no such exclusion. Asymmetric risk allocation heavily favors Vendor.",
        ),
        Clause(
            clause_id="C002-09",
            clause_type="Termination for Convenience",
            text="Either party may terminate this Agreement for convenience upon sixty (60) days written notice. Upon termination, Customer shall pay all fees due through the end of the current Subscription Term.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.TERMINATION,
            is_missing_protection=False,
            annotation="HIGH RISK: Customer must pay fees through end of term even if they terminate early. This effectively eliminates the termination for convenience right. Missing: pro-rata refund on prepaid fees.",
        ),
        Clause(
            clause_id="C002-10",
            clause_type="Data Return",
            text="Upon termination, Vendor will make Customer Data available for export for thirty (30) days, after which Vendor may delete all Customer Data. Export functionality is provided 'as-is'.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.DATA_PRIVACY,
            is_missing_protection=True,
            annotation="'As-is' export is risky — no guarantee data is complete or usable format. 30 days may be insufficient for large datasets. Missing: certification of deletion after the 30-day window.",
        ),
    ]
)

# CONTRACT 3 — EMPLOYMENT AGREEMENT (Hard Task — Adversarial Buried Clause)
EMPLOYMENT_CONTRACT = Contract(
    contract_id="C-003",
    contract_type="Employment",
    title="Executive Employment Agreement",
    parties={"employer": "Nexus Technologies Inc", "employee": "Jordan R. Chen"},
    clauses=[
        Clause(
            clause_id="C003-01",
            clause_type="Compensation",
            text="Employee shall receive an annual base salary of $220,000, payable in accordance with the Company's standard payroll schedule, subject to applicable withholdings and deductions as determined by the Company from time to time.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.AMBIGUOUS_LANGUAGE,
            is_missing_protection=False,
            annotation="'As determined by the Company from time to time' on deductions is vague and could permit non-standard deductions. Should specify deductions are limited to legally required ones.",
        ),
        Clause(
            clause_id="C003-02",
            clause_type="Intellectual Property Assignment",
            text="Employee agrees that all inventions, discoveries, improvements, and works of authorship conceived or developed by Employee during employment, whether or not during working hours or using Company resources, shall be the sole property of the Company and Employee hereby irrevocably assigns all rights thereto.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.IP_OWNERSHIP,
            is_missing_protection=False,
            annotation="HIGH RISK: Assignment covers inventions outside working hours and without Company resources. This is overly broad — many states (CA, DE, IL) limit such assignments to work-related inventions. Employee's personal side projects may be captured.",
        ),
        Clause(
            clause_id="C003-03",
            clause_type="Non-Compete",
            text="For a period of two (2) years following termination for any reason, Employee shall not directly or indirectly engage in any business activity that competes with the Company's business in any geographic area where the Company operates or plans to operate.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.NON_COMPETE,
            is_missing_protection=False,
            annotation="HIGH RISK: 2-year non-compete with unlimited geographic scope ('plans to operate') is overbroad and likely unenforceable in CA, ND, MN, and increasingly other states. No compensation specified during non-compete period.",
        ),
        Clause(
            clause_id="C003-04",
            clause_type="At-Will Employment",
            text="Employment is at-will and may be terminated by either party at any time with or without cause or notice, subject to any severance provisions herein.",
            risk_level=RiskLevel.LOW,
            risk_type=RiskType.TERMINATION,
            is_missing_protection=False,
            annotation="Standard at-will clause. Acceptable if severance provisions are fair.",
        ),
        Clause(
            clause_id="C003-05",
            clause_type="Severance",
            text="In the event of termination without cause, Employee shall receive continuation of base salary for a period of four (4) weeks for each year of service, not to exceed twenty-six (26) weeks total, conditioned upon execution of a general release of all claims against the Company.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.TERMINATION,
            is_missing_protection=False,
            annotation="Severance conditioned on release of claims is standard but should include a review period (21 days for ADEA waiver if over 40). Missing: COBRA continuation or health insurance during severance.",
        ),
        Clause(
            clause_id="C003-06",
            clause_type="Dispute Resolution",
            text="The parties agree that any disputes arising from or related to this Agreement or Employee's employment shall be resolved through final and binding arbitration pursuant to the Company's Arbitration Policy, which the Company may update from time to time at its sole discretion. Employee waives any right to a jury trial. Class action waiver: Employee waives the right to participate in any class or collective action proceeding.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.DISPUTE_RESOLUTION,
            is_missing_protection=False,
            annotation="HIGH RISK: Company can unilaterally change the Arbitration Policy. Class action waiver limits recovery for systemic violations (wage theft, discrimination). Jury trial waiver in an adhesion employment contract may be unconscionable in some states.",
        ),
        Clause(
            clause_id="C003-07",
            clause_type="Garden Leave / Non-Solicitation",
            text="Notwithstanding the foregoing provisions and subject to applicable law, including without limitation those provisions set forth in Section 12(b)(iii) as amended from time to time, the Employee covenants and agrees that during the Restricted Period (as defined in Exhibit A, incorporated herein by reference and subject to modifications as the Company may determine appropriate in its reasonable business judgment), Employee shall not solicit, induce, recruit, or encourage any person who is or was an employee or contractor of the Company within the preceding twenty-four (24) months to terminate their relationship with the Company, whether directly or indirectly, in any capacity whatsoever.",
            risk_level=RiskLevel.HIGH,
            risk_type=RiskType.NON_COMPETE,
            is_missing_protection=False,
            is_buried=True,
            annotation="ADVERSARIAL BURIED CLAUSE: 24-month non-solicitation of employees buried in dense cross-referencing legalese. 'Restricted Period' is defined in Exhibit A (not shown) — scope is deliberately obscured. Effective career limiter.",
        ),
        Clause(
            clause_id="C003-08",
            clause_type="Governing Law",
            text="This Agreement shall be governed by the laws of the State of Texas. Any arbitration shall be conducted in Austin, Texas.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.GOVERNING_LAW,
            is_missing_protection=False,
            annotation="If employee works remotely in a state with stronger employee protections (e.g. CA), Texas governing law may strip those protections. Courts in employee's home state may refuse to enforce.",
        ),
        Clause(
            clause_id="C003-09",
            clause_type="Equity Vesting",
            text="Employee shall be granted options to purchase 50,000 shares subject to a four-year vesting schedule with a one-year cliff. Upon termination for any reason, unvested options shall be immediately forfeited and vested options must be exercised within thirty (30) days of termination.",
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.TERMINATION,
            is_missing_protection=True,
            annotation="30-day exercise window post-termination is too short for ISOs (tax planning needed). Missing: acceleration on change-of-control (single or double trigger). No provision for extended exercise window (some companies offer 5-10 years).",
        ),
        Clause(
            clause_id="C003-10",
            clause_type="Amendment",
            text="This Agreement may only be amended by written instrument signed by both parties.",
            risk_level=RiskLevel.NONE,
            risk_type=RiskType.CLEAN,
            is_missing_protection=False,
            annotation="Standard mutual amendment clause. Clean.",
        ),
    ]
)

ALL_CONTRACTS = {
    "C-001": NDA_CONTRACT,
    "C-002": SAAS_CONTRACT,
    "C-003": EMPLOYMENT_CONTRACT,
}
