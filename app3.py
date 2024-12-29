import streamlit as st       # for web applications
import pickle                # for loading the resume
import docx                  # Extract text from Word file
import PyPDF2                # Extract text from PDF
import re
import nltk


# backend-library
nltk.download('punkt')
nltk.download('stopwords')

# loading models --->
svm = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tf = pickle.load(open('tf.pkl', 'rb'))   # Example file name, adjust as needed

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = None
    
    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
            return None

        # Clean and vectorize the text
        if text:
            st.info("Text extraction successful. Proceeding with vectorization...")
            cleaned_text = cleanResume(text)
            vectorized_text = tf.transform([cleaned_text])
            return vectorized_text, text  # Return both the vectorized data and raw text
        else:
            st.error("Text extraction failed.")
            return None, None
        
        # Summarize the resume text using Hugging Face
        summarized_text = summarize_resume(text)
        return summarized_text
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None, None
    
# Function to calculate ATS score
def calculate_ats_score(resume_text, category_name):
    """
    Calculate ATS score by matching resume text with predefined category keywords.
    Args:
        resume_text (str): The text content of the resume.
        category_name (str): The predicted category name.
    Returns:
        float: ATS score as a percentage.
    """
    # Define keywords for each category
    keyword_dict = {
        "Data Science": ["machine learning", "python", "data analysis", "statistics", "deep learning", "pandas", "numpy"],
        "HR": ["recruitment", "employee engagement", "payroll", "training", "performance management"],
        "Advocate": ["litigation", "legal", "contract", "compliance", "court", "law"],
        "Arts": ["design", "creative", "painting", "drawing", "sculpture", "visual"],
        "Web Designing": ["html", "css", "javascript", "ui", "ux", "responsive", "bootstrap"],
        "Mechanical Engineer": ["cad", "solidworks", "thermodynamics", "manufacturing", "automation"],
        "Sales": ["lead generation", "salesforce", "customer relationship", "negotiation", "target achievement"],
        "Health and fitness": ["nutrition", "exercise", "personal trainer", "fitness plan"],
        "Civil Engineer": ["construction", "autocad", "surveying", "structural design", "geotechnical"],
        "Java Developer": ["java", "spring", "hibernate", "microservices", "jpa"],
        "Business Analyst": ["requirements gathering", "gap analysis", "stakeholders", "agile", "scrum"],
        "SAP Developer": ["sap", "abap", "hana", "erp", "sap modules"],
        "Automation Testing": ["selenium", "test automation", "junit", "pytest", "testing framework"],
        "Electrical Engineering": ["circuit design", "power systems", "embedded systems", "pcb", "electronics"],
        "Operations Manager": ["operations", "logistics", "supply chain", "inventory", "management"],
        "Python Developer": ["python", "django", "flask", "machine learning", "api development"],
        "DevOps Engineer": ["devops", "docker", "kubernetes", "ci/cd", "ansible", "jenkins"],
        "Network Security Engineer": ["firewall", "vpn", "network security", "penetration testing", "encryption"],
        "PMO": ["project management", "stakeholders", "pmo", "budget", "planning"],
        "Database": ["sql", "database management", "mysql", "mongodb", "oracle"],
        "Hadoop": ["hadoop", "big data", "hdfs", "mapreduce", "spark"],
        "ETL Developer": ["etl", "data pipeline", "sql", "data warehouse", "informatica"],
        "DotNet Developer": [".net", "c#", "asp.net", "mvc", "entity framework"],
        "Blockchain": ["blockchain", "smart contract", "ethereum", "solidity", "decentralized"],
        "Testing": ["manual testing", "automation testing", "qa", "bug tracking", "selenium"]
    }

    # Get keywords for the predicted category
    keywords = keyword_dict.get(category_name, [])
    if not keywords:
        return 0.0  # No keywords for this category

    # Normalize resume text
    resume_text = resume_text.lower()

    # Match keywords in the resume text
    keyword_matches = sum(1 for keyword in keywords if keyword.lower() in resume_text)
    ats_score = (keyword_matches / len(keywords)) * 100 if keywords else 0.0

    return round(ats_score, 2)

# Web app ---> creating web application using Streamlit
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    
    st.title("Resume Screening Detector with ATS Score")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        vectorized_resume, resume_text = handle_file_upload(uploaded_file)
        if vectorized_resume is None:
            return  # Stop if processing failed

        st.success("Resume successfully vectorized.")

        # Display extracted text (optional)
        if st.checkbox("Show extracted text", False):
            st.text_area("Extracted Resume Text", resume_text, height=300)  # Show raw text

        # Predict the category
        try:
            prediction_id = svm.predict(vectorized_resume)[0]
            st.write(prediction_id)

            # Map category ID to category name
            category_mapping = {
                6: 'Data Science',
                12: 'HR',
                0: 'Advocate',
                1: 'Arts',
                24: 'Web Designing',
                16: 'Mechanical Engineer',
                22: 'Sales',
                14: 'Health and fitness',
                5: 'Civil Engineer',
                15: 'Java Developer',
                4: 'Business Analyst',
                21: 'SAP Developer',
                2: 'Automation Testing',
                11: 'Electrical Engineering',
                18: 'Operations Manager',
                20: 'Python Developer',
                8: 'DevOps Engineer',
                17: 'Network Security Engineer',
                19: 'PMO',
                7: 'Database',
                13: 'Hadoop',
                10: 'ETL Developer',
                9: 'DotNet Developer',
                3: 'Blockchain',
                23: 'Testing',
            }
            category_name = category_mapping.get(prediction_id, 'Unknown')
            st.success(f"Predicted Category: {category_name}")

            # Calculate ATS score
            ats_score = calculate_ats_score(resume_text, category_name)
            st.info(f"ATS Score: {ats_score}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
