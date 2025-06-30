import os
import json
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Set this in your environment or .env file
# os.environ["OPENAI_API_KEY"] = "sk-..."

class RequestType(str, Enum):
    PRESCRIPTION_REFILL = "prescription_refill"
    APPOINTMENT = "appointment"
    MEDICATION_QUESTION = "medication_question"
    SIDE_EFFECT_REPORT = "side_effect_report"
    GENERAL_INQUIRY = "general_inquiry"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"

class MedicationEntity(BaseModel):
    name: str = Field(description="Name of medication")
    dosage: Optional[str] = Field(None, description="Dosage (e.g. 500mg)")
    frequency: Optional[str] = Field(None, description="How often taken (e.g. 'twice daily')")
    quantity: Optional[str] = Field(None, description="Amount requested")
    
class PersonEntity(BaseModel):
    name: str = Field(description="Full name of person")
    relation: Optional[str] = Field(None, description="Relationship to caller (e.g. 'mother', 'self')")
    
class AddressEntity(BaseModel):
    street: Optional[str] = Field(None, description="Street name and number")
    city: Optional[str] = Field(None, description="City name")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    
class AppointmentEntity(BaseModel):
    date: Optional[str] = Field(None, description="Requested date")
    time: Optional[str] = Field(None, description="Requested time")
    reason: Optional[str] = Field(None, description="Reason for appointment")
    doctor: Optional[str] = Field(None, description="Requested doctor name")

class ParsedRequest(BaseModel):
    request_type: RequestType = Field(description="The type of request being made")
    medications: List[MedicationEntity] = Field(default_factory=list, description="Medications mentioned")
    people: List[PersonEntity] = Field(default_factory=list, description="People mentioned")
    address: Optional[AddressEntity] = Field(None, description="Address information")
    appointment: Optional[AppointmentEntity] = Field(None, description="Appointment information")
    urgency: int = Field(1, description="Urgency level from 1-5, with 5 being most urgent")
    missing_info: List[str] = Field(default_factory=list, description="List of important missing information")
    language: str = Field("de", description="Detected language of the request")
    confidence: float = Field(0.0, description="Confidence score (0.0-1.0) of the extraction")
    raw_text: str = Field("", description="Original text that was analyzed")
    
    @validator('urgency')
    def validate_urgency(cls, v):
        if not (1 <= v <= 5):
            raise ValueError("Urgency must be between 1 and 5")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


def analyze_user_input(
    user_text: str, 
    model: str = "gpt-4o", 
    temperature: float = 0.0
) -> ParsedRequest:
    """
    Analyzes transcribed user input to extract structured information.
    
    Args:
        user_text: The transcribed text from the user
        model: LLM model to use
        temperature: Temperature setting for generation (0.0 = deterministic)
        
    Returns:
        Structured information extracted from the text
    """
    # Define system prompt with detailed instructions
    system_prompt = """
    You are an AI assistant for a German doctor's office and pharmacy.
    Your task is to precisely analyze patient calls and extract structured information.
    
    IMPORTANT INSTRUCTIONS:
    - The input will be in German or occasionally English
    - Extract ALL relevant medical entities (medications, dosages, etc.)
    - Extract ALL mentioned people and their relationships
    - Determine the exact request type
    - Assess urgency on a scale of 1-5 (5 = emergency)
    - List any critical missing information
    - Format your response as a valid JSON object matching the specified schema
    - Be precise and thorough - this information will be used for medical purposes
    
    For medications, capture:
    - Exact medication names (even with spelling errors)
    - Dosage information (strength, frequency, quantity)
    
    For people, capture:
    - Full names
    - Relationships to caller
    
    For addresses, capture any mentioned:
    - Street names and numbers
    - Cities
    - Postal codes
    
    For appointments, capture:
    - Dates
    - Times
    - Reasons
    - Specific doctor requests
    
    Rate urgency:
    1 = routine request
    2 = needs attention within days
    3 = needs attention within 24 hours
    4 = needs immediate attention
    5 = emergency situation requiring immediate escalation
    
    You MUST return a valid JSON object according to the specified schema.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_text}")
    ])
    
    # Create a parser for JSON output
    parser = JsonOutputParser(pydantic_object=ParsedRequest)
    
    # Create the processing chain
    chain = (
        prompt 
        | ChatOpenAI(model=model, temperature=temperature) 
        | parser
    )
    
    # Process the user text
    try:
        result = chain.invoke({"user_text": user_text})
        # Add the original text to the result
        result['raw_text'] = user_text
        return ParsedRequest(**result)
    except Exception as e:
        # Fallback for parsing errors
        return ParsedRequest(
            request_type=RequestType.UNKNOWN,
            missing_info=["Failed to parse request completely"],
            confidence=0.1,
            raw_text=user_text,
            language="unknown"
        )


def format_parsed_request(parsed: ParsedRequest) -> str:
    """Format the parsed request as a human-readable string."""
    output = []
    output.append(f"Request Type: {parsed.request_type.value}")
    output.append(f"Urgency Level: {parsed.urgency}/5")
    output.append(f"Confidence: {parsed.confidence:.2f}")
    
    if parsed.medications:
        output.append("\nMedications:")
        for med in parsed.medications:
            med_info = f"- {med.name}"
            if med.dosage:
                med_info += f" ({med.dosage})"
            if med.frequency:
                med_info += f", {med.frequency}"
            if med.quantity:
                med_info += f", Quantity: {med.quantity}"
            output.append(med_info)
    
    if parsed.people:
        output.append("\nPeople:")
        for person in parsed.people:
            person_info = f"- {person.name}"
            if person.relation:
                person_info += f" ({person.relation})"
            output.append(person_info)
    
    if parsed.address and any([parsed.address.street, parsed.address.city, parsed.address.postal_code]):
        output.append("\nAddress:")
        address_parts = []
        if parsed.address.street:
            address_parts.append(f"Street: {parsed.address.street}")
        if parsed.address.city:
            address_parts.append(f"City: {parsed.address.city}")
        if parsed.address.postal_code:
            address_parts.append(f"Postal Code: {parsed.address.postal_code}")
        output.append("- " + ", ".join(address_parts))
    
    if parsed.appointment and any([parsed.appointment.date, parsed.appointment.time, 
                                   parsed.appointment.reason, parsed.appointment.doctor]):
        output.append("\nAppointment:")
        if parsed.appointment.date:
            output.append(f"- Date: {parsed.appointment.date}")
        if parsed.appointment.time:
            output.append(f"- Time: {parsed.appointment.time}")
        if parsed.appointment.reason:
            output.append(f"- Reason: {parsed.appointment.reason}")
        if parsed.appointment.doctor:
            output.append(f"- Doctor: {parsed.appointment.doctor}")
    
    if parsed.missing_info:
        output.append("\nMissing Information:")
        for info in parsed.missing_info:
            output.append(f"- {info}")
            
    return "\n".join(output)


# Example usage
if __name__ == "__main__":
    # Test with various scenarios
    test_inputs = [
        "Hallo, ich würde gerne Ibuprofen 600mg für meine Mutter Maria Schmidt bestellen bitte.",
        "Guten Tag, ich brauche einen Termin bei Dr. Mueller nächste Woche Freitag wegen meiner Rückenschmerzen.",
        "Ich habe starke Brustschmerzen und kann kaum atmen. Was soll ich tun?",
        "Hallo, ich nehme Metformin zweimal täglich und habe bemerkt, dass ich Magenschmerzen bekomme. Ist das normal?",
    ]
    
    for text in test_inputs:
        print("\n" + "="*50)
        print(f"INPUT: {text}")
        print("="*50)
        
        result = analyze_user_input(text)
        
        # Print formatted result
        print(format_parsed_request(result))
        
        # Also show the raw JSON for debugging
        print("\nRAW JSON:")
        print(json.dumps(result.dict(), indent=2, ensure_ascii=False))