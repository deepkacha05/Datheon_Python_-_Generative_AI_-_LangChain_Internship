"""Pydantic schemas for Task 1: Structured LLM Prompt Pipeline & JSON Extractor.

Defines strongly-typed data contracts for extracting structured information
from unstructured texts like Resumes and Invoices.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# =====================================================================
# 1. Resume Extraction Schemas
# =====================================================================

class ContactInfo(BaseModel):
    """Candidate contact information."""
    name: Optional[str] = Field(default=None, description="Candidate full name if present in contact block")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number with country code if available")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(default=None, description="GitHub profile URL")
    location: Optional[str] = Field(default=None, description="City, State, or Country of residence")


class Education(BaseModel):
    """Educational qualification details."""
    institution: str = Field(description="University, college, or school name")
    degree: str = Field(description="Degree or diploma earned, e.g., B.Tech, B.S., M.S.")
    field_of_study: Optional[str] = Field(default=None, description="Major/Field of study, e.g., Computer Science")
    graduation_year: Optional[int] = Field(default=None, description="Year of graduation or expected completion")
    gpa: Optional[str] = Field(default=None, description="GPA or percentage score, e.g., '3.8/4.0' or '85%'")


class WorkExperience(BaseModel):
    """Professional work experience item."""
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title or designation")
    location: Optional[str] = Field(default=None, description="Job location or 'Remote'")
    start_date: Optional[str] = Field(default=None, description="Start date (e.g., 'Jan 2022' or '2022-01')")
    end_date: Optional[str] = Field(default=None, description="End date (e.g., 'Dec 2023', 'Present')")
    is_current: bool = Field(default=False, description="Whether this is the current active role")
    responsibilities: List[str] = Field(default_factory=list, description="Key contributions and bullet points")
    technologies: List[str] = Field(default_factory=list, description="Tech stack / tools utilized in this role")


class Project(BaseModel):
    """Academic or personal project details."""
    name: str = Field(description="Project name or title")
    description: Optional[str] = Field(default=None, description="Brief summary of what the project does")
    technologies: List[str] = Field(default_factory=list, description="Libraries, tools, and languages used")
    link: Optional[str] = Field(default=None, description="GitHub repository or live URL")


class ResumeSchema(BaseModel):
    """Complete structured representation of a parsed resume."""
    candidate_name: str = Field(description="Full legal name of the candidate")
    headline: Optional[str] = Field(default=None, description="Professional headline or title, e.g., 'AI Engineer'")
    contact_info: ContactInfo = Field(default_factory=ContactInfo, description="Contact and social details")
    summary: Optional[str] = Field(default=None, description="Professional summary or career objective")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills list")
    work_experience: List[WorkExperience] = Field(default_factory=list, description="Chronological work history")
    education: List[Education] = Field(default_factory=list, description="Degrees and educational credentials")
    projects: List[Project] = Field(default_factory=list, description="Key technical projects")
    certifications: List[str] = Field(default_factory=list, description="Certificates and awards")

    @field_validator("candidate_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("candidate_name cannot be empty")
        return clean


# =====================================================================
# 2. Invoice Extraction Schemas
# =====================================================================

class InvoiceParty(BaseModel):
    """Entity details for vendor or customer/client."""
    name: str = Field(description="Company or individual name")
    address: Optional[str] = Field(default=None, description="Physical billing or shipping address")
    email: Optional[str] = Field(default=None, description="Contact email address")
    phone: Optional[str] = Field(default=None, description="Contact telephone number")
    tax_id: Optional[str] = Field(default=None, description="Tax ID, VAT, GSTIN, or EIN")


class InvoiceLineItem(BaseModel):
    """Single billable item or service entry in an invoice."""
    description: str = Field(description="Item name or service description")
    quantity: float = Field(default=1.0, description="Quantity or hours billed")
    unit_price: float = Field(default=0.0, description="Price per unit")
    total: float = Field(default=0.0, description="Total line cost (qty * unit_price)")

    @field_validator("quantity", "unit_price", "total", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        if isinstance(v, str):
            # Strip currency symbols, commas, spaces
            cleaned = v.replace("$", "").replace("₹", "").replace("€", "").replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return v


class InvoiceSchema(BaseModel):
    """Complete structured representation of a commercial invoice."""
    invoice_number: str = Field(description="Unique invoice identifier or code")
    invoice_date: Optional[str] = Field(default=None, description="Date invoice was issued (YYYY-MM-DD or as text)")
    due_date: Optional[str] = Field(default=None, description="Payment due date")
    vendor: InvoiceParty = Field(description="Seller or service provider details")
    client: InvoiceParty = Field(description="Buyer or client receiving the invoice")
    line_items: List[InvoiceLineItem] = Field(default_factory=list, description="List of items billed")
    subtotal: float = Field(default=0.0, description="Gross amount before taxes and discounts")
    tax_rate_percent: Optional[float] = Field(default=None, description="Tax rate percentage if noted")
    tax_amount: float = Field(default=0.0, description="Calculated tax amount")
    discount_amount: float = Field(default=0.0, description="Discount amount applied")
    total_amount: float = Field(description="Final payable amount")
    currency: str = Field(default="USD", description="Currency code, e.g., USD, INR, EUR")
    payment_terms: Optional[str] = Field(default=None, description="Payment terms, e.g., 'Net 30'")
    notes: Optional[str] = Field(default=None, description="Additional notes, bank instructions, or footer")

    @field_validator("subtotal", "tax_amount", "discount_amount", "total_amount", mode="before")
    @classmethod
    def coerce_amounts(cls, v):
        if isinstance(v, str):
            cleaned = v.replace("$", "").replace("₹", "").replace("€", "").replace(",", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return v
