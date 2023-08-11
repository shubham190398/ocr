import pypdfium2 as pdfium
import cv2
import os
import easyocr
import time
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import csv


def cheque_transcribe(img):
    pass


def invoice_transcribe(img):
    pass


def main():
    pdf_dir = os.listdir('../dataset/bad_img_pdfs')

    for file in pdf_dir:

        pdf = pdfium.PdfDocument('../dataset/bad_img_pdfs/' + file)
        print(len(pdf))

        cheque_pdf = pdf[0]
        invoice_pdf = pdf[-1]

        cheque = cheque_pdf.render().to_numpy()
        invoice = invoice_pdf.render().to_numpy()


if __name__ == '__main__':
    main()