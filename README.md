# Code_Fun_Do_PlusPlus
The Challenge - Predict,Detect or Manage Natural Disasters

Stark101's Proposal:

Usually, it takes a lot of time for the disaster management authorities to be notified about any natural disaster that might have taken place. 

For example: https://www.usgs.gov/faqs/how-fast-does-earthquake-information-get-posted-web-site-get-sent-out-earthquake-notification?qt-news_science_products=0#qt-news_science_products
suggests that it might take upto 20 minutes in some areas to know whether there has been an earthquaqe in that part of the world. In this project we aim to provide a solution to the same.

The live feed provided by the government owned surveilance tools (cameras/satelite data) are periodically scanned frame wise. The frames captured are checked for any anomalies. If any serious event is detected, the concerned authorities are immediately pinged to look into the situation, and take the required action.

Since this is done periodically (say every 30 seconds), the communication gap is bridged between the concerned authorities and the line of action and quick relief can be brought to the affected area.

The second feature of our project is to Predict Natural Disasters. This is done by capturing animal movements in a zoo as it is common knowledge that animals have a tendency to show erratic behaviour a few days before a calamity. This is done through live cctv surveilance of animals in zoos in terms of them coming in and out of their respective caves and then storing that information into a database. This data is analysed to observe anomalies and give the concerned authorities time to inform and prepare the public about the for a natural disaster.

We at Stark101 sincerely hope to implement this solution because this methodology can bring about quick relief where every minute is precious.

YouTube: https://www.youtube.com/watch?v=2GOf97N8iLM&t=34s

To run: python3 frontend.py

Note -> A additional file credentials.py has to be created by the person using this program,wherein the details of the access of the access 
Username, Password, Database name, driver and server name has to be provided. We are not providing our own details for security reasons. 
