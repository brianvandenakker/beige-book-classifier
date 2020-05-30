# -*- coding: utf-8 -*-
"""
Purpose: Scrape and format the Federal Reserves Beige book data for text analysis
Inputs: The page_root for the Federal Reserve URL
Outputs: A dataframe of the data by Beigebook subsection
"""

#%%
#Function: yearURL()
#Purpose: Returns the specified URL for a year in the Federal Reserves Beige book archives.
#Inputs: A year in which data is available
#Output: The URL which has links to the ~8 Beige books written prior to the FOMC meetings
def yearURL(year):
    global yearDict
    global page_root
    yearDict = dict()
    page_root = 'https://www.federalreserve.gov'
    page_link = os.path.join(page_root,"monetarypolicy/beige-book-archive.htm").replace("\\", "/")
    #Request archive url
    page_response = requests.get(page_link, timeout=5)
    #Fetch archive url content
    page_content = BeautifulSoup(page_response.content, "html.parser")
    #Isolate links
    list_links = page_content.find('div', {'class' : 'panel-body'}).find_all('li')
    #Define regular expression
    re_link = re.compile('<a href="(?P<nextlevel>.*)".*')
    #Loop through each link to access month url
    for i in range(0, len(list_links)):
        #Isolate and clean individual links
        link = str(list_links[i]).replace('\n','').replace('<li>','').replace('</li>','')
        #Extract next level
        nllink = re_link.match(link).groupdict()['nextlevel']
        #Define year level URL
        yearlink = os.path.join(page_root, nllink.strip('/')).replace('\\', '/')
        yearDict[int(re.findall(r'\d+', yearlink)[0])] = [yearlink]
    #Return specified year URL
    if(1995 < year < 2019):
        return yearDict[year][0]
    else: print("That data is not available. The archives are currently for 1996 - 2018")
        
    
#%%   
    
#Function: meetingURL()
#Purpose: Returns a dictionary or URL's for each of the Beige books written in a given year.
#Inputs: The year in which we are interested in doing analysis.
#Output: The URL's to access the individual Beige books.
def meetingURL(year):
    root = yearURL(year)
    global meetingDict
    meetingDict = dict()
    #Scrape content from yearURL
    #Request URL
    year_response = requests.get(root, timeout=5)
    #Scrape year URL content
    year_content = BeautifulSoup(year_response.content, "html.parser")
    #Isolate meeting specific paths
    meeting = year_content.find('div', {'class' : 'col-xs-12 col-sm-8 col-md-8'})
    meeting_links = meeting.find_all('a')
    #Define regular expression
    re_link = re.compile('<a href="(?P<nextlevel>.*)".*')
    #Delay web scrape
    time.sleep(1.5)
    #Loop through each month in which there was a meeting per year
    for x in range(0,len(meeting_links)):
        if(('www.federalreserve.gov' in str(meeting_links[x]) or '/monetarypolicy/beigebook' in str(meeting_links[x])) and '.htm' in str(meeting_links[x])):
            meetinglink = re_link.match(str(meeting_links[x])).groupdict()['nextlevel']
            if('www.federalreserve.gov/fomc' in str(meeting_links[x])):
                meetingDict[int(re.findall(r'\d+', meetinglink)[1])] = [meetinglink]
            elif('www.federalreserve.gov' in str(meeting_links[x])):
                meetingDict[int(re.findall(r'\d+', meetinglink)[0])] = [meetinglink]
            else:
                meetingDict[int(re.findall(r'\d+', meetinglink)[0])] = [os.path.join(page_root, meetinglink.strip('/')).replace('\\', '/')]    
    if(1995 < year < 2019):
        return meetingDict
    else: print("That data is not available. The archives are currently for 1996 - 2018")


#%%
    
#Data Scraping:   
#Note: The sites have three different formats: (1) 1996 - 2010, (2) 2011 - 2016, & (3) 2017-2018  
#Must utilize individual functions to handle these different formats. 

#Functions: scrapeOne(), scrapeTwo(), scrapeThree()
#Purpose: Scrape text data from specific Beige books
#Inputs: A year and meeting number (1-8), due to 8 meetings per year (only 2 in 1996)
#Outputs: The text contents for that Beige book
    

#1996 - 2010
def scrapeOne(year, meeting):
    global text
    #Call the function to return updated meetingDict for required year
    meetingURL(year)
    #Identify specific meeting URL
    textURL = meetingDict[list(meetingDict.keys())[meeting - 1]][0]
    textURL = textURL.replace('default', 'FullReport')
    #Request the URL
    text_response = requests.get(textURL, timeout = 5)
    #Scrape the contents
    text = BeautifulSoup(text_response.content, "html.parser")
    return text

#2011 - 2016
def scrapeTwo(year, meeting):
    global text
    #Call the function to return updated meetingDict for required year
    meetingURL(year)
    #Identify specific meeting URL
    textURL = meetingDict[list(meetingDict.keys())[meeting - 1]][0]
    #Request the URL
    text_response = requests.get(textURL, timeout = 5)
    #Scrape the contents
    text_content = BeautifulSoup(text_response.content, "html.parser")
    #Full report
    text = text_content.find('div', {'id' : 'leftText'})
    return text


#2017 - 2018
def scrapeThree(year, meeting):
    global text
    #Call the function to return updated meetingDict for required year
    meetingURL(year)
    #Identify specific meeting URL
    textURL = meetingDict[list(meetingDict.keys())[meeting - 1]][0]
    #Request the URL
    text_response = requests.get(textURL, timeout = 5)
    #Scrape the contents
    text_content = BeautifulSoup(text_response.content, "html.parser")
    #Full report
    text = text_content.find('div', {'id' : 'article'})
    return text


#%%
    
#Function: warning()
#Purpose: Let's user know if their inputs are invalid
#Inputs: year and FOMC meeting number for that year.
#Outputs: A warning if the inputs are invalid given the data, else it passes.
def warning(year, meeting):
    if(meeting < 1 or meeting > 8):
        print("There are 8 FOMC meetings per year")
        return True
    elif(year == 1996 and meeting > 2):
        print("We only have data for the last two meetings in 1996")
        return True
    elif(year < 1996 or year > 2018):
        print("We don't have data for this year")
        return True
    

#Function: scrapeText()
#Purpose: General purpose function combining all scraping techniques utilized in the previous functions
#Inputs: Year and FOMC meeting number for that year
#Outputs: The text contained in the specified book
def scrapeText(year, meeting):
    global text
    if(warning(year, meeting) != True):
        if(year < 2011):
            return scrapeOne(year, meeting)
        elif(year > 2016):
            return scrapeThree(year, meeting)
        else: return scrapeTwo(year, meeting)


#%%
#Scrape text data to scrapedDataList
data = []
for i in range(1996, 2019):
    for j in range(1, len(meetingURL(i))+1): 
        print(i)
        print(j)
        x = scrapeText(i, j)
        y = {'year':i, 'meeting':j,'text':x}
        data.append(y.copy())


#%%
#Borrowed from: https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
#Strips HTML tags from strings
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
#%%
  

for index, data in enumerate(data):
    data[index]['text'] = strip_tags(str(data[index]['text'])).replace('\r', '').replace('\n', '')


raw_text = os.path.join(savedDataFld, "raw_text.json").replace('\\', '/')
with open(raw_text, 'w') as f:
    json.dump(data, f)



#%%
