'''
This module contains helperfunctions to load pdfs, extract their texts and generate additional metadata

It was initially created for the businessresponsibility.ch project of the Prototype Fund. For more
information visit https://github.com/bizres

'''

from datetime import datetime

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

import langid
langid.set_languages(['en', 'de','fr','it'])

import pandas as pd

def get_pdf_last_modified(path):
    '''
    This function gets the 'ModDate' field from a pdf

    Parameters:
    path: Path to pdf
    '''
    try:
        with open(path, 'rb') as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            date = datetime.strptime(doc.info[0]['ModDate'].decode("utf-8")[2:10], '%Y%m%d')
            return date
    except:
        return None

def pdf_to_text(path):
    '''
    This function extracts text from a pdf. Pages are separatet with \n\n, sections are separatet wth \n

    Parameters:
    path: path to pdf
    '''
    with open(path, 'rb') as file:
        try:
            full_text = ''

            # Iterate over every page of document
            for page_layout in extract_pages(file):

                page_text = ''

                # Iterate over every container on page
                for element in page_layout:

                    # Only consider containers which are marked as text containers (no headers, footers, tables or images)
                    if isinstance(element, LTTextContainer):

                        section_text = ''

                        # Iterate over every line of text container
                        for text_line in element:
                            section_text += text_line.get_text().replace('\n', ' ')
                                
                        # Remove all whitespaces (except ' ') such as new lines or tabs
                        section_text = ' '.join(section_text.split())

                        page_text += section_text # Add section text to page text
                        page_text += '\n' # Separate sections with a new line

                full_text += page_text # Add page text to full text
                full_text += '\n' # Separate pages with a double new line

        except:
            full_text = ''
        
        return full_text

def detect_language(text):
    '''
    This function detects the language of a text using langid
    '''
    return langid.classify(text)

def pdf_text_to_sections(text):
    '''
    This function generates a pandas DataFrame from the extracted text. Each section
    is provided with the page it is on and a section_index
    '''
    sections = []
    page_nr = 0
    section_index = 0
    for page in text.split('\n\n'):
        page_nr += 1
        for section in page.split('\n'):
            sections.append([page_nr, section_index, section])
            section_index += 1

    return pd.DataFrame(sections, columns=['page', 'section_index', 'section_text'])
