import fitz  # PyMuPDF
from transformers import pipeline
import spacy
import os
import re
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import tkinter as tk
from tkinter import filedialog
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, Settings, StorageContext,Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pyvis.network import Network
from IPython.display import display, HTML
