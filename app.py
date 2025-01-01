from flask import Flask, Response, request, url_for, redirect, session, render_template
from pymongo import MongoClient
from datetime import datetime
import uuid
from urllib.parse import parse_qs
from html import unescape

from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "testingupdaterssdita"

conn = MongoClient('mongodb+srv://widyandita:cheesetoast@cluster0.vdchw.mongodb.net/')
db = conn.get_database('test_data_berita')
users_collection = db.test_users

def default_news_data():
    collection_1 = db.test_berita1
    data = list(collection_1.find())
    dfdata = pd.DataFrame(data)
    dfdata['_id'] = dfdata['_id'].astype(str)

    return dfdata

def generate_recommendations(tracked_news_id):
    df1 = default_news_data()
    clicked_id = tracked_news_id
    rec_num = 5

    tfidf_matrix = TfidfVectorizer()
    cleanedt = df1['Cleaned_Title'].values
    clickedt = df1['Cleaned_Title'][df1['_id'] == clicked_id]
    tfidf_matrix.fit(cleanedt)
    tfidf_vector = tfidf_matrix.transform(cleanedt)
    i_tfidf_vector = tfidf_matrix.transform(clickedt)

    similarity = cosine_similarity(i_tfidf_vector, tfidf_vector)
    rank = similarity.flatten().argsort()[::-1]
    final_recommended_articles_index = [article_id for article_id in rank if article_id not in [0]][:rec_num]
    filtereddf = df1.loc[final_recommended_articles_index]
    recsids = filtereddf['_id'].values

    return recsids    

@app.route("/")
def welcome():
    session.clear()
    # Generate a new user ID
    user_id = str(uuid.uuid4())
    
    # Store the user in the database
    users_collection.insert_one({"user_id": user_id, "created_at": datetime.utcnow()})

    session["user_id"] = user_id
    
    # Generate the RSS feed URL for the user
    rss_url = url_for("generate_rss_feed", user_id=user_id, _external=True)
    
    # Render the welcome page with the custom RSS URL
    return render_template(
        'welcomepg.html',
        rss_url = rss_url
    )

@app.route("/rss", methods=["GET"])
def generate_rss_feed():
    def rfc822_date():
        now = datetime.utcnow()
        return now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    user_id = session.get("user_id")
    
    # Check if the user exists in the database
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return "Invalid user ID", 404

    tracked_news_id = session.get("tracked_news_id", None)

    # print(user_id)
    # print(tracked_news_id)
    
    if tracked_news_id is not None:
        # Generate recommendations based on the tracked ID
        recommended_ids = generate_recommendations(tracked_news_id)
    else:
        # Default to including all news items in no specific order
        recommended_ids = []

    # Fetch all news data from MongoDB
    all_news_df = default_news_data()
    
    # Ensure pub_date is a datetime object
    all_news_df["pub_date"] = pd.to_datetime(all_news_df["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")

    # Sort DataFrame by pub_date in descending order
    sorted_news_df = all_news_df.sort_values(by="pub_date", ascending=False)
    sorted_news_df = sorted_news_df.to_dict(orient="records")

    all_news_dict = {news["_id"]: news for news in sorted_news_df}  # Map for quick lookup by ID

    # Arrange items based on recommendation order
    ordered_news = [all_news_dict[news_id] for news_id in recommended_ids if news_id in all_news_dict]

    # Include any remaining items that weren't in the recommendations
    remaining_news = [news for news in sorted_news_df if news["_id"] not in recommended_ids]
    ordered_news.extend(remaining_news)

    # Begin manual RSS construction
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    title = SubElement(channel, "title")
    title.text = f"Custom News Feed for User {user_id}"
    link = SubElement(channel, "link")
    link.text = request.url_root
    description = SubElement(channel, "description")
    description.text = "Latest news updates."
    last_build_date = SubElement(channel, "lastBuildDate")
    last_build_date.text = rfc822_date()

    # Add items in the correct order
    for news in ordered_news:
        item = SubElement(channel, "item")
        item_title = SubElement(item, "title")
        item_title.text = news["Title"]
        item_link = SubElement(item, "link")
        item_link.text = url_for("track", user_id=user_id, news_id=news["_id"], redirect="true", _external=True)
        item_guid = SubElement(item, "guid")
        item_guid.text = item_link.text

        pub_date_str = news["pub_date"]
        
        formatted_date = pub_date_str.strftime("%a, %d %b %Y %H:%M:%S +0000")
        pubdate = SubElement(item, "pubDate")
        pubdate.text = formatted_date

    # Generate and prettify XML
    rough_string = tostring(rss, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_rss_feed = reparsed.toprettyxml(indent="  ")

    session.pop("tracked_news_id", None)

    # print(ordered_news)  # Debugging output for order verification
    # print(tracked_news_id)

    return Response(pretty_rss_feed, content_type="application/rss+xml")

@app.route("/track", methods=["GET"])
def track():
    raw_query_string = request.query_string.decode("utf-8")
    query_params = parse_qs(unescape(raw_query_string))  # Decode and parse

    user_id = query_params.get("user_id", [None])[0]
    news_id = query_params.get("news_id", [None])[0]
    
    # print("Raw Query String:", raw_query_string)
    # print("Decoded Query String:", query_params)

    redirect_to = request.args.get("redirect", "true")  # Extract 'redirect' parameter
    news_data = default_news_data().to_dict(orient="records")

    # print("User ID:", user_id)
    # print("News ID:", news_id)
    # print("Redirect Parameter:", redirect_to)

    # Validate parameters
    if redirect_to == "true" and news_id is not None:
        news_item = next((item for item in news_data if str(item["_id"]) == str(news_id)), None)
        if not news_item:
            return "News item not found", 404

        original_url = news_item["Link"]

        # Log the click
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")        
        collection_2 = db.test_log3
        collection_2.insert_one({
        "user_id": user_id,
        "news_id": news_id,
        "timestamp": timestamp
        })
              
        session["tracked_news_id"] = news_id
        # print(f"Tracked news ID saved in session: {news_id}")

        return redirect(original_url)
    
    return "Invalid tracking parameters", 400

if __name__ == "__main__":
    app.run()