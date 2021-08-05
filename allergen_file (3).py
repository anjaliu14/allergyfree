
### IMPORT STATEMENTS #######
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import time
import requests
from bs4 import BeautifulSoup
import re
#coding=utf-8
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import ast

from PIL import Image
import re
import unicodedata

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import pipeline
import pickle
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import json


###### WEBSCRAPING FUNCTIONS ########
def fraction_finder(s):
  new_string = ""
  for c in s:
      try:
          name = unicodedata.name(c)
      except ValueError:
          continue
      if name.startswith('VULGAR FRACTION'):
          normalized = unicodedata.normalize('NFKC', c)
          numerator, slash, denominator = normalized.partition('⁄')
          new_string += str(numerator) + '/' + str(denominator)
      else:
          new_string += str(c)
  return new_string

def scrape_yummly(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')

  yummly_amounts = []
  yummly_units = []
  yummly_ing = []
  yummly_notes = []
  for elem in soup.find_all('li', class_="IngredientLine"):
      a = elem.find('span', class_='amount')
      u = elem.find('span', class_='unit')
      i = elem.find('span', class_='ingredient')
      n = elem.find('span', class_='remainder')
      if a:
          yummly_amounts.append(a.get_text())
      else:
          yummly_amounts.append("")

      if u:
          yummly_units.append(u.get_text())
      else:
          yummly_units.append("")

      if i:
          yummly_ing.append(i.get_text())
      else:
          yummly_ing.append("")

      if n:
          yummly_notes.append(n.get_text())
      else:
          yummly_notes.append("")

  return yummly_amounts, yummly_units, yummly_ing, yummly_notes

def clean_yummly(amounts, units, ing, notes):
  # remove first element of lists because it's a title
  if "recipe" in notes[0].lower():
      amounts = amounts[1:]
      units = units[1:]
      ing = ing[1:]
      notes = notes[1:]

  # remove \xa0 in amount strings
  amounts = [a.replace(u'\xa0', u'') for a in amounts]

  # lower case ingredients
  ing = [i.lower().strip() for i in ing]

  return amounts, units, ing, notes

def full_yummly(url):
  amounts, units, ing, notes = scrape_yummly(url)
  amounts_cleaned, units_cleaned, ing_cleaned, notes_cleaned = clean_yummly(amounts, units, ing, notes)
  return amounts_cleaned, units_cleaned, ing_cleaned

def scrape_delish(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')

  delish_amounts = []
  delish_units = []
  delish_descriptions = []
  for elem in soup.find_all('div', class_="ingredient-item"):
      a = elem.find('span', class_='ingredient-amount')
      d = elem.find('span', class_='ingredient-description')
      if a:
          num_unit = a.get_text()
          split_list = num_unit.split('\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t')
          delish_amounts.append(split_list[0].strip())
          delish_units.append(split_list[1].strip())
      else:
          delish_amounts.append("")
          delish_units.append("")
      if d:
          delish_descriptions.append(d.get_text())
      else:
          delish_descriptions.append("")

  return delish_amounts, delish_units, delish_descriptions


def clean_delish(amounts, units, ingredients):
  temp = []
  notes = []

  # remove \xa0
  ingredients = [i.replace(u'\xa0', u' ') for i in ingredients]

  for i in ingredients:
      stripped_ing = i.strip()
      note_pattern0 = '\([a-z0-9,\.\s\/]*\)'
      no_paren_comment = re.sub(note_pattern0, '$$$', stripped_ing)
      temp += [no_paren_comment.strip().lower()]

  # take out comments (), adjectives
  temp1 = []
  note_pattern = ''
  for i in range(len(temp)):
      # don't remove certain words that might match note pattern 1
  #         note_pattern0 = '^(?!.*jelly).*$' #|.*STRING2|.*STRING3
      # extract and append the note to notes
      note_pattern1 = '^[a-z]+ked$|[a-z]+ted|[a-z]+ped|[a-z]+ned|[a-z]+ced|[a-z]+xed|[a-z]+ved|[a-z]+less|^[a-z]+ly$'
      # anything after a comma is a comment
      note_pattern2 = '|,[\S\s]+'
      # other adjectives found that can be notes
      note_pattern3 = '|torn|ground|medium|flakey'
      # in case the parentheses get split because of extra spaces - if one parentheses attached - comment
      note_pattern4 = '|(\([a-z0-9,\.\s\/]*|[a-z0-9,\.\s\/]*\))'
      note_pattern += note_pattern1 + note_pattern2 + note_pattern3 + note_pattern4

      comment = re.sub(note_pattern, '$$$', temp[i])
      # we want to take out the things that are $$$ or have $$$ and spaces (no other words in it)
      remove_dollar_pattern = '^[^a-z]+$'

      if not re.findall(remove_dollar_pattern, comment):
          if '$$$' in comment:
              temp1.append(re.sub('\$\$\$', '', comment))
          else:
              temp1.append(comment)

  ingredients = [i.strip() for i in temp1]

  return amounts, units, ingredients

def full_delish(url):
  amounts, units, ingredients = scrape_delish(url)
  amounts_cleaned, units_cleaned, ing_cleaned = clean_delish(amounts, units, ingredients)
  return amounts_cleaned, units_cleaned, ing_cleaned

def scrape_tasty(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')

  ing_html = soup.find_all('li', class_="ingredient xs-mb1 xs-mt0")
  ing = [i.get_text("<!-- -->") for i in ing_html]

  return ing

def extract_tasty(ing):
  # lowercase each string and remove empty strings from ingredList
  ingredList = [s.lower() for s in ing if s != '']

  nums = []
  units = []
  foods = []
  notes = []

  for i in range(len(ingredList)):
      # remove parentheses and everything after comma
      note_pattern = '\([a-z0-9\!\<\>\-\.\s\/]*\)'
      note_pattern2 = '|,[\S\s]+'
      note_pattern += note_pattern2
      result = re.findall(note_pattern + note_pattern2, ingredList[i])
      if result:
          notes.append(' '.join(result).strip())
      else:
          notes.append('')

      ingredList[i] = re.sub(note_pattern, '', ingredList[i])


      # remove whitespace and <!-- --> on either end
      ingredList[i] = ingredList[i].strip(' <!->')

      # weird encoding - might have to do this for other common fractions
      ingredList[i] = ingredList[i].replace('½', '1/2')
      ingredList[i] = ingredList[i].replace('¾', '3/4')
      ingredList[i] = ingredList[i].replace('¼', '1/4')

      # extract and append the number to nums
      num_pattern = '^(\d+\/*\.*\d*)([to\-\s]*\d+\/*\.*\d*)*'
      result = re.search(num_pattern, ingredList[i])
      if result:
          nums.append(result.group(0).strip())
      else:
          nums.append('')

      # remove the number from the original string
      ingredList[i] = re.sub(num_pattern, '', ingredList[i])

      # split on remaining <!-- -->
      unit_ing = ingredList[i].split('<!-- -->')
      if len(unit_ing) == 2:
          units.append(unit_ing[0].strip())
          foods.append(unit_ing[1].strip())
      elif len(unit_ing) == 1:
          units.append("")
          foods.append(unit_ing[0].strip())


  return nums, units, foods

def full_tasty(url):
  ingredients = scrape_tasty(url)
  amounts_cleaned, units_cleaned, ing_cleaned = extract_tasty(ingredients)
  return amounts_cleaned, units_cleaned, ing_cleaned

def scrape(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')

  # QUANTITIES
  quantities_html = soup.find_all('div', class_="recipe-ingredients__ingredient-quantity")
  quantities = [i.get_text() for i in quantities_html]

  # INGREDIENTS
  ingredients_html = soup.find_all('span', class_="recipe-ingredients__ingredient-part")
  ingredients = [i.get_text() for i in ingredients_html]

  return quantities, ingredients

def extract(quantities, ingredients):
  # separating units from rest of string
  sep = "   "
  temp = []
  notes = []
  for i in ingredients:
      stripped_ing = i.strip()
      note_pattern0 = '\([a-z0-9,\.\s\/]*\)'
      result = re.findall(note_pattern0, stripped_ing)
      if result:
          notes.append(' '.join(result).strip().replace(",", ""))
      else:
          notes.append('')

      no_paren_comment = re.sub(note_pattern0, '$$$', stripped_ing)


      and_or_phrase_pattern = "((or|and)+\s\s\s+)|(\s\s\s+(or|and)+)"
      res = re.findall(and_or_phrase_pattern, no_paren_comment)
      print(res, no_paren_comment)
      if res:
          temp += [no_paren_comment.strip().lower()]
      else:
          temp += [x.strip().lower() for x in no_paren_comment.split(sep)]



  # take out comments (), adjectives

  temp1 = []
  for i in range(len(temp)):
      # extract and append the note to notes
      note_pattern = '^[a-z]+ked$|[a-z]+ted|[a-z]+ped|[a-z]+ned|[a-z]+ced|[a-z]+xed|[a-z]+ved|[a-z]+less|[a-z]+ly'
      # anything after a comma is a comment
      note_pattern2 = '|,[\S\s]+'
      # other adjectives found that can be notes
      note_pattern3 = '|torn|ground|medium|flakey'
      # in case the parentheses get split because of extra spaces - if one parentheses attached - comment
      note_pattern4 = '|(\([a-z0-9,\.\s\/]*|[a-z0-9,\.\s\/]*\))'
      note_pattern += note_pattern2 + note_pattern3 + note_pattern4
      #'|,[a-z0-9\.\s\/\(\)]+'


      result = re.findall(note_pattern, temp[i])
      if result:
          notes.append(' '.join(result).strip().replace(",", ""))
      else:
          notes.append('')

      comment = re.sub(note_pattern, '$$$', temp[i])

      # we want to take out the things that are $$$ or have $$$ and spaces (no other words in it)
      remove_dollar_pattern = '^[^a-z]+$'

      if not re.findall(remove_dollar_pattern, comment):
          if '$$$' in comment:
              temp1.append(re.sub('\$\$\$', '', comment))
          else:
              temp1.append(comment)

  # ACCOUNT FOR EMPTY STRING THAT ARE IN QUANTITIES - translate that to empty strings in units
  # find indices of empty strings in quantities list (have that as a list)
  # multiply list by 2
  print(quantities, temp1)
  q_inds = [i*2 for i, val in enumerate(quantities) if val == ""]

  # add empty strings at those indices
  for new_ind in q_inds:
      if temp1[new_ind] != "":
          temp1.insert(new_ind, "")


  # separating into units and food
  print("BEFORE SEP", temp1)
  food = []
  units = []
  for i in range(len(temp1)):
      if i % 2 == 0:
          units.append(temp1[i].strip())
      else:
          food.append(temp1[i].strip())


  return quantities, notes, units, food

def full_food(url):
  quantities, ingredients = scrape(url)
  amounts_cleaned, notes_cleaned, units_cleaned, ing_cleaned = extract(quantities, ingredients)
  return amounts_cleaned, units_cleaned, ing_cleaned

def scrape_foodnetwork(link):
  page = requests.get(link)
  soup = BeautifulSoup(page.content, 'html.parser')
  ingredients_html = soup.find_all('p', class_="o-Ingredients__a-Ingredient")
  ingredients_text = [i.get_text() for i in ingredients_html]
  #st.write(ingredients_text)
  return ingredients_text

def clean_foodnetwork(text):
  #st.write(text)
  text.remove('\n\n\n\nDeselect All\n\n')
  for i in range(len(text)):
      text[i] = re.sub('\\n', '', text[i]).lower()
  return text

def extract_foodnetwork(text):
  quantities = []
  notes = []
  units = []
  foods = []
  for i in range(len(text)):
      if "recipe follows" in text[i]:
          continue

      quant_pattern = "(^(\d+\/*\.*\d*)([to\-\s]*\d+\/*\.*\d*)*|^one)"
      result = re.search(quant_pattern, text[i])
      if result:
          quantities.append(result.group(0))
      else:
          quantities.append("")
      text[i] = re.sub(quant_pattern, "", text[i])

      text[i] = text[i].strip()

      note_pattern = "(\([a-z0-9\.\s\/\-',;]*\)|, for serving|, optional|, such as.+$)"
      result = re.findall(note_pattern, text[i])
      if result:
          notes.append(" ".join(result))
      else:
          notes.append("")
      text[i] = re.sub(note_pattern, "", text[i])

      text[i] = text[i].strip()

      unit_patterns = ['stalk', 'medium', 'large', 'clove', 'wedge', 'bag', 'cup', 'teaspoon', 'ounce', 'tablespoon', 'package', 'pound', 'piece', 'slice', 'can', 'halves', 'half', 'box', 'boxes']
      unit_pattern = '(' + 's*|'.join(unit_patterns) + ')'
      result = re.findall(unit_pattern, text[i])
      if result:
          units.append(" ".join(result))
      else:
          units.append("")
      text[i] = re.sub(unit_pattern, "", text[i])

      text[i] = text[i].strip()

      foods.append(text[i])

  return quantities, notes, units, foods

def full_foodnetwork(link):
  ingredients_text = scrape_foodnetwork(link)
  cleaned_text = clean_foodnetwork(ingredients_text)
  quantities, notes, units, foods = extract_foodnetwork(cleaned_text)
  return quantities, units, foods, notes

def scrape_allrecipes(link):
  page = requests.get(link)
  soup = BeautifulSoup(page.content, 'html.parser')
  ingredients_html = soup.find_all('li', class_="ingredients-item")
  ingredients_text = [i.get_text() for i in ingredients_html]
  return ingredients_text

def clean_allrecipes(text):
  for i in range(len(text)):
      text[i] = re.sub('\\n', '', text[i])
      text[i] = fraction_finder(text[i])
      text[i] = text[i].replace('\u2009', ' ')
      text[i] = text[i].strip().lower()
  return text

def extract_allrecipes(text):
  quantities = []
  notes = []
  units = []
  foods = []
  for i in range(len(text)):
      if "recipe follows" in text[i]:
          continue

      quant_pattern = "(^(\d+\/*\.*\d*)([to\-\s]*\d+\/*\.*\d*)*|^one)"
      result = re.search(quant_pattern, text[i])
      if result:
          quantities.append(result.group(0))
      else:
          quantities.append("")
      text[i] = re.sub(quant_pattern, "", text[i])

      text[i] = text[i].strip()

      note_pattern = "(\([a-z0-9\.\s\/\-',;]*\)|, for serving|, optional|, such as.+$|, or as needed|to taste)"
      result = re.findall(note_pattern, text[i])
      if result:
          notes.append(" ".join(result))
      else:
          notes.append("")
      text[i] = re.sub(note_pattern, "", text[i])

      text[i] = text[i].strip()

      unit_patterns = ['stalk', 'medium', 'large', 'clove', 'wedge', 'bag', 'cup', 'teaspoon', 'ounce', 'tablespoon', 'package', 'pound', 'piece', 'slice', 'can', 'halves', 'half', 'box', 'boxes']
      unit_pattern = '(' + 's*|'.join(unit_patterns) + ')'
      result = re.findall(unit_pattern, text[i])
      if result:
          units.append(" ".join(result))
      else:
          units.append("")
      text[i] = re.sub(unit_pattern, "", text[i])

      text[i] = text[i].strip()

      foods.append(text[i])

  return quantities, notes, units, foods

def full_allrecipes(link):
  ingredients_text = scrape_allrecipes(link)
  cleaned_text = clean_allrecipes(ingredients_text)
  quantities, notes, units, foods = extract_allrecipes(cleaned_text)
  return quantities, units, foods, notes

def all_scrape(url):
  cut_url = url.lower()
  #     print(cut_url.split('/'))

  if cut_url[0:4] == "http":
      cut_url = cut_url.split('/')[2]
  else:
      cut_url = cut_url.split('/')[0]
  #     print(cut_url)

  if "yummly" in cut_url:
      return full_yummly(url)

  elif "delish" in cut_url:
      return full_delish(url)

  elif "tasty" in cut_url:
      return full_tasty(url)

  elif "foodnetwork" in cut_url:
      return full_foodnetwork(url)

  elif "food" in cut_url: # put food network before food
      return full_food(url)

  elif "allrecipes" in cut_url:
      return full_allrecipes(url)

  # else: handle manual input (user enters recipe manually)



#@st.cache()


####Sidebar and Initial Layout######
rad = st.sidebar.radio("Navigation", ["Home", "About", "Meet the Creators"])
#st.sidebar.image("/content/circle-cropped (1).png")



##### CLASSIFY FUNCTIONS (URL & MANUAL) #####
def classify(url, allergens):
  #allergen must be in list format: Example: [dairy, tree nuts] or [dairy]
  #options for allergen: ['dairy', 'eggs', 'tree nuts', 'peanuts', 'gluten', 'seafood', 'soy']
  ingredient_list = all_scrape(url)[2]
  count = 0
  allergen_data = pd.read_csv("/content/drive/MyDrive/allergen_free_materials_v2/allergen_master_w_joined_word.csv")
  aller_ing = []
  ing_aller_dict = dict()
  for allergen in allergens:
      for ingredient in ingredient_list:
          if any(food in ingredient for food in allergen_data[allergen_data[allergen] == 1]['food'].to_list()):
              ing = "".join(ingredient.strip().split())
              ing_aller_dict[ing] = allergen
              aller_ing.append(ingredient)
              count += 1
      #this sometimes doesn't show up for more than two allergens - need to double check this
      if count == 0:
          st.write("This recipe is " + allergen + " free!") #outputs the ingredient and allergen class on streamlit
  # for i in ingredient_list:
  #   st.write(i)
  ingredient_list = ["".join(i.strip().split()) for i in ingredient_list]
  return ingredient_list, ing_aller_dict
  

def classify_manual(ingredient_list, allergens):
  ingredients= ingredient_list
  count = 0
  allergen_data = pd.read_csv("/content/drive/MyDrive/allergen_free_materials_v2/allergen_master_w_joined_word.csv")
  aller_ing = []
  ing_aller_dict = dict()
  for allergen in allergens:
      for ingredient in ingredients:
          if any(food in ingredient for food in allergen_data[allergen_data[allergen] == 1]['food'].to_list()):
              ing = "".join(ingredient.strip().split())
              ing_aller_dict[ing] = allergen
              aller_ing.append(ingredient)
              count += 1
      if count == 0:
          #print("This recipe is" + allergen + " free!")
          st.write("This recipe is " + allergen + " free!")
  ingredients = ["".join(i.strip().split()) for i in ingredients]
  # for i in ingredients:
  #   st.write(i)
  #st.write(ing_aller_dict)
  return ingredients, ing_aller_dict
   



#From knn_allergy_function.ipynb
def get_word_idx(sent: str, word: str):
  return sent.split(" ").index(word)

def get_hidden_states(encoded, token_ids_word, model, layers):
  """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
  with torch.no_grad():
      output = model(**encoded)

  # Get all hidden states
  states = output.hidden_states
  # Stack and sum all requested layers
  output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
  # Only select the tokens that constitute the requested word
  word_tokens_output = output[token_ids_word]

  return word_tokens_output.mean(dim=0)
 
 
def get_word_vector(sent, idx, tokenizer, model, layers):
  """Get a word vector by first tokenizing the input sentence, getting all token idxs
  that make up the word of interest, and then `get_hidden_states`."""
  encoded = tokenizer.encode_plus(sent, return_tensors="pt")
  # print(encoded)
  # print(encoded.encoded)
  # print("HELLO")
  # get all token idxs that belong to the word of interest
  token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
  
  return get_hidden_states(encoded, token_ids_word, model, layers)


def get_word_embeddings(model, tokenizer, sent, word, layers=None):
  # Use last four layers by default
  layers = [-4, -3, -2, -1] if layers is None else layers
  
  idx = get_word_idx(sent, word)

  word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)

  return word_embedding

def string_to_list(string_list):
  return ast.literal_eval(string_list)

def add_space_commas(recipe_lst):
  return " , ".join(recipe_lst)

def knn_recipes(word_embed_df, ingred_vector, neigh):
  """
  word_embed_df is a dataframe
  ingred_vector is a vector of the ingredient we want predictions for
  """

  ingred_vector = ingred_vector.tolist()
  # list looks like [[490 437 164 882 411 755 395 863 810 866]]
  k_neighbors_list = neigh.kneighbors([ingred_vector], return_distance=False)

  k_neighbors_ingredients = []
  for i in k_neighbors_list[0]:
    # print(vector_representations.iloc[i]["ingredients"])
    k_neighbors_ingredients.append(word_embed_df['ingredients_space'][str(i)])
  
  return k_neighbors_ingredients


def ing_choose_substitutions(neighbors, orig_ing, allergen):
  counter = 0 
  final_subs = []
  allergen_data = pd.read_csv("/content/drive/MyDrive/allergen_free_materials_v2/allergen_master_w_joined_word.csv")
  for sub in neighbors:
    sub = sub.strip() # some ingredients have extra spaces in front
    if any(food in sub for food in allergen_data[allergen_data[allergen] == 1]['food'].to_list()):
      pass
    else:
      if counter < 3:
        if sub not in final_subs:
          final_subs.append(sub)
          print(sub + ': does NOT contain ' + allergen)
          counter += 1
      else:
        return final_subs
  return final_subs



def show_substitions(ingredient_list, ing_aller_dict):
#This function, when called, will output the substitions on streamlit based on the saved model, word embedddings, knn output 

#Models and Code for Displaying the Substitutions  
  tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/model_save_3")
  model = AutoModel.from_pretrained("/content/drive/MyDrive/model_save_3", output_hidden_states=True)    

  allergen_data = pd.read_csv("/content/drive/MyDrive/allergen_free_materials_v2/allergen_master_w_joined_word.csv")
  #word_embeddings = pd.read_csv("/content/drive/MyDrive/allergen_free_materials_v2/full_ingred_embeddings.csv")
  #word_embedddings = pd.read_json("/content/drive/MyDrive/allergen_free_materials_v2/full_embeddings_columns.json")
  #word_embeddings["ingred_list"] = word_embeddings["ingred_list"].apply(string_to_list)
  #word_embeddings["embeddings_list"] = word_embeddings["embeddings_list"].apply(string_to_list) 
  # Opening JSON file
  f = open("/content/drive/MyDrive/allergen_free_materials_v2/full_embeddings_columns.json",)
  # returns JSON object as 
  # a dictionary
  word_embeddings = json.load(f)
  # Closing file
  f.close()
  #remove embeddings column from words_embeddings

  # ###### THIS CAN POTENTIALLY BE DONE OUTSIDE SO WE AREN'T DOING THIS EVERYTIME AND FITTING
  vect_list_train = list(word_embeddings["embeddings_list"].values())
  neigh = NearestNeighbors(n_neighbors=10000) #perhaps consider decreasing neighbors for faster performance
  neigh.fit(vect_list_train)
  # ##################

  #### TESTING
  recipe_sent = add_space_commas(ingredient_list)
  #st.write(recipe_sent)

  # for orig_ing, allergen_group in ing_aller_dict.items():
  #   #print("SUBSTITUTIONS FOR ", orig_ing, " (allergen group is ", allergen_group, ")")
  #   md = f"**{orig_ing}** contains **{allergen_group}**."
  #   st.markdown(md, unsafe_allow_html=True)
  #   orig_ing = orig_ing.replace(" ", "")
  #   ing_word_embed = get_word_embeddings(model, tokenizer, recipe_sent, orig_ing)
  #   ing_neighbors = knn_recipes(word_embeddings, ing_word_embed, neigh)
  #   ing_choose_substitutions(ing_neighbors, orig_ing, allergen_group)
  #   #print()

  for orig_ing, allergen_group in ing_aller_dict.items():
    md = f"**{orig_ing}** contains **{allergen_group}**."
    st.markdown(md, unsafe_allow_html=True)
    #md2 = f"POSSIBLE SUBSTITUTIONS FOR **{orig_ing}**:"
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Possible substitutions include:**")
    #orig_ing = orig_ing.replace(" ", "")
    #print("SUBSTITUTIONS FOR ", orig_ing, " (allergen group is ", allergen_group, ")")
    #if orig_ing in word_embeddings['ingredients_space'].to_list():
    ing_word_embed = get_word_embeddings(model, tokenizer, recipe_sent, orig_ing)
    ing_neighbors = knn_recipes(word_embeddings, ing_word_embed, neigh)
    lst_final = ing_choose_substitutions(ing_neighbors, orig_ing, allergen_group)
    if len(lst_final) > 0:
      for s in lst_final:
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- {s}")
    else:
      st.write("Sorry! No substitutions available for this ingredient.")


def main():


    html_temp = """

    <div style ="background-color:pink;padding:13px">

    <h1 style ="color:black;text-align:center;">Allergy Free  </h1>


    </div>

    """


 

    # display the front end aspect

    # st.markdown(html_temp, unsafe_allow_html = True)
    st.image("/content/circle-cropped (1).png") 

    # page_bg_img = '''
    # <style>
    # body {
    # background-image: url("hhttps://previews.123rf.com/images/romastudio/romastudio1603/romastudio160300287/54494047-healthy-food-background-studio-photography-of-different-fruits-and-vegetables-on-white-wooden-backgr.jpg");
    # background-size: cover;
    # }
    # </style>
    # '''
    #
    # st.markdown(page_bg_img, unsafe_allow_html=True)

    #https://i.pinimg.com/originals/a1/12/15/a112152ef5b6d8fe425dd3cf0c3c7dd7.jpg
    #https://www.freshvegetablesontario.com/fvgo/assets/img/homepageRight.png
    st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.freshvegetablesontario.com/fvgo/assets/img/homepageRight.png")


    }
    </style>
    """,
    unsafe_allow_html=True
)



    st.subheader('This app supports recipes from the following websites:')
    st.write('- food.com')
    st.write('- yummly.com')
    st.write('- delish.com')
    st.write('- tasty.com')
    st.write('- foodnetwork.com')
    st.write('- allrecipes.com')
    st.write('*If you have a recipe in mind that is not found in a website above, you can enter the recipe manually.*')

    #st.header("Waiting for website to load...")
    st.header("Let's Begin")
    entry = st.radio(
     "Recipe Entry",
     ('URL', 'Manual'))
    if entry == 'URL':
        url = st.text_input('Enter the url for the recipe of your choice: ')
        if st.button("See ingredient details: "):
            # q, i = scrape(url)
            # extracted_info = extract(q, i)
            ingredient_list = all_scrape(url)[2]
            st.write("The recipe you entered calls for: ")
            for ing in ingredient_list:
                st.write(ing)
        corrections = st.radio("Does the above ingredient list look right?", ["yes", "no"])
        if corrections == "no":
            st.write("Oh no! You can re-enter it manually")
            ingredients_string= st.text_input('Add ingredients seperated by comma. Example: milk, salt, butter')
            #if st.button("See ingredient details"):
            ingredients = ingredients_string.split(',')
            ingredients = [ing.strip() for ing in ingredients]
            st.write("The recipe you entered calls for: ")
            for ing in ingredients:
                st.write(ing)
            aller = st.multiselect('Select the allergies that apply to you', ['eggs', 'soy', 'dairy', 'gluten', 'tree nuts', 'peanuts', "seafood"])
            allergens = [a for a in aller]
            if st.button('Give me the substitutions please!'):
                ingredient_list, ing_aller_dict = classify_manual(ingredients, allergens)
                #tokenizer, model, word_embeddings, neigh = initialization()
                show_substitions(ingredient_list, ing_aller_dict)   
        if corrections == 'yes':
            aller = st.multiselect('Select the allergies that apply to you', ['eggs', 'soy', 'dairy', 'gluten', 'tree nuts', 'peanuts', "seafood"])
            allergens = [a for a in aller]
            if st.button('Give me the substitutions please!'):
                ingredient_list, ing_aller_dict = classify(url, allergens)
                #tokenizer, model, word_embeddings, neigh = initialization()
                show_substitions(ingredient_list, ing_aller_dict)
    if entry == 'Manual':
        ingredients_string= st.text_input('Add ingredients seperated by comma. Example: milk, salt, butter')
        ingredients = ingredients_string.split(',')
        ingredients = [ing.strip() for ing in ingredients]
        if st.button("See ingredient details!"):
            st.write("The recipe you entered calls for: ")
            for ing in ingredients:
                st.write(ing)
        aller = st.multiselect('Select the allergies that apply to you', ['eggs', 'soy', 'dairy', 'gluten', 'tree nuts', 'peanuts', "seafood"])
        allergens = [a for a in aller]
        if st.button('Give me the substitutions please!'):
                ingredient_list, ing_aller_dict = classify_manual(ingredients, allergens)
                #tokenizer, model, word_embeddings, neigh = initialization()
                show_substitions(ingredient_list, ing_aller_dict) 





#if __name__=='__main__':

    #main()

if rad=='Home':
  main()
if rad == 'About':
    st.subheader("Why Allergy Free?")
    st.write("Often times, people with allergies or specific dietary restrictions such as having a nut allergy, being vegetarian, or being gluten free have a hard time following recipes for a variety of foods. Allergy Free allows our users to input any recipe from supported recipe websites. Based on the ingredient breakdown of the recipe along with their selected allergies, Allergy Free suggests ingredient substitutions the user could make to their recipe to align with their needs! This allows our users to not have to worry about finding specific recipes for their allergens, as they can still use any recipe they like with substitutions when needed!")
    st.image("https://t4.ftcdn.net/jpg/03/13/43/95/360_F_313439588_W18RtX1Ye1eNnNwp6SXATZHDPxE4bjjy.jpg")
if rad == 'Meet the Creators':
    col1, col2= st.beta_columns(2)
    col3, col4 = st.beta_columns(2)
    anusha = Image.open('/content/drive/MyDrive/allergy_free_materials/Anusha_website.jpg')
    anjali = Image.open('/content/drive/MyDrive/allergy_free_materials/Anjali_website.jpg')
    shalini = Image.open('/content/drive/MyDrive/allergy_free_materials/Shalini_website.jpg')
    lavanya = Image.open('/content/drive/MyDrive/allergy_free_materials/Lavanya_website.jpg')
    col1.header("Anusha Mohan")
    col1.image(anusha, use_column_width=True)
    col1.write("Anusha Mohan is a 5th Year Masters in Information and Data Science student at UC Berkeley and will be graduating in Fall 2021. After graduation, she will be working as a data science analyst at Tatari, a tv marketing analytics company in San Francisco! She enjoys creating models and finding meaningful insights in data that can help improve products or create applications that can benefit others. She is excited about Allergy Free and its ability to make cooking easier and more accessible to everyone. In her free time she loves trying new restaurants, exploring places with friends and family, as well as spending time with her golden retriever!")
    col2.header("Anjali Unnithan")
    col2.image(anjali,use_column_width=True)
    col2.write("Anjali Unnithan is a 5th Year Masters in Information and Data Science student at UC Berkeley and will be graduating in Fall 2021. She currently works as a data scientist at SAP and will continue to do so after graduation. Her interests lie in machine learning applications and using data visualization to tell a story. She hopes for Allergy Free to make a positive impact on many! In her free time she loves listening to music, watching movies, dancing, and traveling!")
    col3.header("Shalini Kunapuli")
    col3.image(shalini,use_column_width=True)
    col3.write("Shalini Kunapuli is a 5th Year Masters of Information and Data Science student at UC Berkeley, graduating in Fall 2021. She recently finished her undergrad at UC Berkeley where she studied Statistics and Computer Science. She is passionate about using data and technology to improve the world, and is a huge proponent of getting more females into computer science and closing the gender gap in tech fields. She is very interested in pursuing Data Science/Analytics in the future, previously she has been a Data Science/Analytics intern at Quizlet, LinkedIn, and UpLift Inc. As someone who has many food allergies, she is extremely excited about Allergy Free and the potential to make cooking and eating more inclusive for people with various dietary restrictions. In her free time, she loves to dance, play the clarinet, read, explore and travel to new places, and vlog/make YouTube videos (she recently started a YouTube channel focused around technology, professional development, women in tech, and more)!")
    col4.header("Lavanya Vijayan")
    col4.image(lavanya,use_column_width=True)
    col4.write("Lavanya Vijayan is a 5th Year Masters in Information and Data Science student at UC Berkeley and will be graduating in Summer 2021. Lavanya is also Founding Lead Instructor & Strategist at Afro Fem Coders, a published author of programming courses on LinkedIn Learning, a coding instructor at The Coder School Berkeley, and a Math & French tutor at Berkeley High School’s Bridge program. She is passionate about improving access to coding and data science education, as well as using data to build tools that are inclusive and accessible. In her free time, she enjoys singing, writing, and exploring new restaurants and cafes!")