from flask import Flask, Response, jsonify, request, url_for, redirect, render_template
import os
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import uuid
from urllib.parse import parse_qs
from html import unescape

from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load environment variables (production values are provided via app.yaml)
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')
app.config['BASE_URL'] = os.getenv('BASE_URL', 'https://widya-nandita-tugas-akhir.et.r.appspot.com')
app.config['PREFERRED_URL_SCHEME'] = 'https'

BASE_URL = "https://widya-nandita-tugas-akhir.et.r.appspot.com"

mongo_uri = os.getenv('MONGO_URI')
conn = MongoClient(mongo_uri)
db = conn.get_database('test_data_berita')
users_collection = db.test_users2

def rfc822_date():
    now = datetime.utcnow()
    return now.strftime("%a, %d %b %Y %H:%M:%S GMT")

def default_news_data():
    collection_1 = db.data_berita
    data = list(collection_1.find())
    dfdata = pd.DataFrame(data)
    dfdata['_id'] = dfdata['_id'].astype(str)

    return dfdata

def first_news_data():
    collection_1 = db.test_data_first
    data = list(collection_1.find())
    dfdata = pd.DataFrame(data)
    dfdata['_id'] = dfdata['_id'].astype(str)

    return dfdata

def generate_avg_recommendations(user_id):
    clicked_news = users_collection.find_one({"user_id": user_id}, {"clicked_news": 1})

    sorted_news_ids = []
    if clicked_news and "clicked_news" in clicked_news:
        sorted_news_ids = [
            news["news_id"] for news in sorted(clicked_news["clicked_news"], key=lambda x: x["timestamp"])
        ]
   
    unique_news_ids = []
    for item in sorted_news_ids:
        if item not in unique_news_ids:
            unique_news_ids.append(item)
    
    df1 = default_news_data()
    rec_num = 5

    tfidf_matrix = TfidfVectorizer()
    cleanedt = df1['Cleaned_Title'].values
    filtered_df = df1[df1['_id'].isin(unique_news_ids)]
    # Determine the indices of the clicked news in the DataFrame
    clicked_indices = df1.index[df1['_id'].isin(unique_news_ids)].tolist()
    # Extract the title values as a list
    titles = filtered_df['Cleaned_Title'].tolist()

    tfidf_matrix.fit(cleanedt)
    tfidf_vector = tfidf_matrix.transform(cleanedt)
    i_tfidf_vector = tfidf_matrix.transform(titles)
    composite_vector = np.mean(i_tfidf_vector.toarray(), axis=0).reshape(1, -1)

    similarity = cosine_similarity(composite_vector, tfidf_vector)
    rank = similarity.flatten().argsort()[::-1]
    final_recommended_articles_index = [
                                        i for i in rank if i not in clicked_indices
                                    ][:rec_num]
    filtereddf = df1.loc[final_recommended_articles_index]
    recsids = filtereddf['_id'].values
    recsids = recsids.tolist()

    return recsids

def generate_default_rss_feed(user_id, default_news):
    # Begin manual RSS construction
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    title = SubElement(channel, "title")
    title.text = f"RSS News Feed Yang Dikustomisasi Untuk User {user_id}"
    link = SubElement(channel, "link")
    #link.text = request.url_root
    link.text = BASE_URL
    language = SubElement(channel, "language")
    language.text = "id"
    generator = SubElement(channel, "generator")
    generator.text = "Custom RSS Feed Generator with xml.etree.ElementTree"
    copyright = SubElement(channel, "copyright")
    copyright.text = "Copyright © 2025 CustomizedRSS. All rights reserved."
    description = SubElement(channel, "description")
    description.text = "Kumpulan berita terbaru dari berbagai situs berita Indonesia. Silahkan membaca beberapa berita untuk mendapatkan rekomendasi berita."
    last_build_date = SubElement(channel, "lastBuildDate")
    last_build_date.text = rfc822_date()

    # Add items in the correct order
    for news in default_news:
        item = SubElement(channel, "item")
        item_title = SubElement(item, "title")
        item_title.text = news["Title"]
        item_description = SubElement(item, "description")
        item_description.text = news["Description"]
        item_link = SubElement(item, "link")
        item_link.text = url_for("track", user_id=user_id, news_id=news["_id"], redirect="true", _external=True)
        item_guid = SubElement(item, "guid", attrib={"isPermaLink": "false"})
        item_guid.text = item_link.text

        pub_date_str = news["pub_date"]
        
        formatted_date = pub_date_str.strftime("%a, %d %b %Y %H:%M:%S +0000")
        pubdate = SubElement(item, "pubDate")
        pubdate.text = formatted_date

    # Generate and prettify XML
    rough_string = tostring(rss, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pi = reparsed.createProcessingInstruction("xml-stylesheet", 'type="text/xsl" href="style.xsl"')
    reparsed.insertBefore(pi, reparsed.firstChild)
    pretty_rss_feed = reparsed.toprettyxml(indent="  ")

    return pretty_rss_feed

def generate_recs_time(df1,df2):
    df1["pub_date"] = pd.to_datetime(df1["pub_date"])
    df2["pub_date"] = pd.to_datetime(df2["pub_date"])

    # Get the last time from df1 and latest from df2
    last_time_1 = df1["pub_date"].min()
    latest_time_2 = df2["pub_date"].max()

    interval = (latest_time_2 - last_time_1) / 6

    timestamps = [last_time_1 + i * interval for i in range(1,6)]

    return timestamps

@app.route("/")
def welcome():
    # Generate a new user ID
    user_id = str(uuid.uuid4())

    # Fetch first news data and title from MongoDB
    first_news_df = first_news_data()
    filter_titles = first_news_df["Title"]

    # Ensure pub_date is a datetime object and convert to dict
    first_news_df["pub_date"] = pd.to_datetime(first_news_df["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")
    first_news_dict = first_news_df.to_dict(orient="records")

    # Fetch all news data from MongoDB
    all_news_df = default_news_data()
    all_news_df["pub_date"] = pd.to_datetime(all_news_df["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")
    news_dict = all_news_df.to_dict(orient="records")

    # Sort DataFrame by pub_date in descending order and filter by 4 days
    sorted_news_df = all_news_df.sort_values(by="pub_date", ascending=False)
    sorted_news_df["date_only"] = sorted_news_df["pub_date"].dt.date
    recent_dates = sorted_news_df["date_only"].drop_duplicates().sort_values(ascending=False).head(4)
    sorted_news_df = sorted_news_df[sorted_news_df["date_only"].isin(recent_dates)]
    sorted_news_df = sorted_news_df.to_dict(orient="records")

    all_news_dict = {news["_id"]: news for news in news_dict}  # Map for quick lookup by ID

    recommended_ids = []

    # Arrange items based on recommendation order
    ordered_news = [all_news_dict[news_id] for news_id in recommended_ids if news_id in all_news_dict]

    first_news_dict.extend(ordered_news)

    # Include any remaining items that weren't in the recommendations
    remaining_news = [
                    news for news in sorted_news_df
                    if news["_id"] not in recommended_ids and news["Title"] not in filter_titles
                ]
    first_news_dict.extend(remaining_news)

    pretty_rss_feed = generate_default_rss_feed(user_id, first_news_dict)
    
    # Store the user in the database
    users_collection.insert_one({
        "user_id": user_id,
        "created_at": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "clicked_news": [],
        "last_recommendation_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "recommendations": [],
        "rss_feed": pretty_rss_feed
    })
    
    # Generate the RSS feed URL for the user
    rss_url = url_for("generate_rss_feed", user_id=user_id, _external=True)
    
    # Render the welcome page with the custom RSS URL
    return render_template(
        'welcomepg.html',
        base_rss_url = rss_url
    )

@app.route("/updaterss", methods=["GET"])
def update_rss_feed():
    users = users_collection.find()
    with app.app_context():
        for user in users:
            user_id = user["user_id"]
            
            # Check if the user exists in the database
            user = users_collection.find_one({"user_id": user_id}, {"clicked_news": 1})
            if not user:
                return "Invalid user ID", 404

            if user and "clicked_news" in user:
            # Sort the clicked_news by timestamp and extract the news_id
                sorted_news_ids = [
                    news["news_id"] for news in sorted(user["clicked_news"], key=lambda x: x["timestamp"])
                ]

            if len(sorted_news_ids) > 0:
                # Generate recommendations based on the tracked ID
                recommended_ids = generate_avg_recommendations(user_id)
            else:
                # Default to including all news items in no specific order
                recommended_ids = []

            rec_time = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"recommendations": recommended_ids,
                        "last_recommendation_time": rec_time}
                }
            )
        
        return jsonify({"message": "News recommendations updated for all users"})

@app.route("/rss", methods=["GET"])
def generate_rss_feed():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    sort_flag = request.args.get("sort", "false").lower() == "true"

    # Fetch the user's RSS feed from the database
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return jsonify({"error": "User ID not found"}), 404
    
    if "recommendations" in user:
        recommended_ids = user["recommendations"]
        
    # Fetch first news data and title from MongoDB
    first_news_df = first_news_data()
    filter_titles = first_news_df["Title"]

    # Ensure pub_date is a datetime object and convert to dict
    first_news_df["pub_date"] = pd.to_datetime(first_news_df["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")
    first_news_dict = first_news_df.to_dict(orient="records")

    # Fetch all news data from MongoDB
    all_news_df = default_news_data()
    all_news_df["pub_date"] = pd.to_datetime(all_news_df["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")
    news_dict = all_news_df.to_dict(orient="records")

    # Sort DataFrame by pub_date in descending order and filter by 4 days
    sorted_news_df = all_news_df.sort_values(by="pub_date", ascending=False)
    sorted_news_df["date_only"] = sorted_news_df["pub_date"].dt.date
    recent_dates = sorted_news_df["date_only"].drop_duplicates().sort_values(ascending=False).head(4)
    sorted_news_df = sorted_news_df[sorted_news_df["date_only"].isin(recent_dates)]
    sorted_news_df = sorted_news_df.to_dict(orient="records")

    all_news_dict = {news["_id"]: news for news in news_dict}  # Map for quick lookup by ID

    # Arrange items based on recommendation order
    ordered_news = [all_news_dict[news_id] for news_id in recommended_ids if news_id in all_news_dict]

    if sort_flag:
        new_timestamps = generate_recs_time(first_news_df, all_news_df)

        for news, new_time in zip(ordered_news, new_timestamps):
            news['pub_date'] = new_time 

    first_news_dict.extend(ordered_news)

    # Include any remaining items that weren't in the recommendations
    remaining_news = [
                    news for news in sorted_news_df
                    if news["_id"] not in recommended_ids and news["Title"] not in filter_titles
                ]
    first_news_dict.extend(remaining_news)

    pretty_rss_feed = generate_default_rss_feed(user_id, first_news_dict)

    users_collection.update_one({"user_id": user_id}, {"$set": {"rss_feed": pretty_rss_feed}})
    
    # Return the RSS feed content
    response = Response(pretty_rss_feed, content_type="application/rss+xml")
    return response

@app.route('/uat_recommendations', methods=['GET'])
def get_uat_recommendations():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID not found."})

    # Fetch recommendations from MongoDB
    user_data = users_collection.find_one({'user_id': user_id})
    recommended_ids = user_data["recommendations"]
    
    if user_data and "clicked_news" in user_data:
        # Sort the clicked_news by timestamp and extract the news_id
        clicked_ids = [
            news["news_id"] for news in sorted(user_data["clicked_news"], key=lambda x: x["timestamp"])
        ]
        clicked_titles = [news_title_map[news_id] for news_id in clicked_ids if news_id in news_title_map]
    
    if recommended_ids:
        all_news_df = default_news_data()
        news_title_map = dict(zip(all_news_df["_id"], all_news_df["Title"]))
        ordered_news = [news_title_map[news_id] for news_id in recommended_ids if news_id in news_title_map]
        
    if len(clicked_ids) == 0:
        clicked_titles = "No news were clicked yet."

    if len(recommended_ids) == 0:
        ordered_news = "No recommendations were made yet."

    return jsonify({"clicked_news":clicked_titles,
                    "recommended_news":ordered_news})


@app.route("/track", methods=["GET"])
def track():
    raw_query_string = request.query_string.decode("utf-8")
    query_params = parse_qs(unescape(raw_query_string))  # Decode and parse

    user_id = query_params.get("user_id", [None])[0]
    news_id = query_params.get("news_id", [None])[0]

    redirect_to = request.args.get("redirect", "true")  # Extract 'redirect' parameter
    news_data = first_news_data().to_dict(orient="records")
    all_news_data = default_news_data().to_dict(orient="records")
    news_data.extend(all_news_data)
    
    # Validate parameters
    if redirect_to == "true" and news_id is not None:
        news_item = next((item for item in news_data if str(item["_id"]) == str(news_id)), None)
        if not news_item:
            return "News item not found", 404

        original_url = news_item["Link"]

        # Log the click
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")        
        users_collection.update_one(
        {"user_id": user_id},
        {"$push": {"clicked_news": {
            "news_id": news_id, "timestamp": timestamp
        }}}
        )

        return redirect(original_url)
    
    return "Invalid tracking parameters", 400

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_rss_feed, 'interval', minutes=50)
    scheduler.start()

    app.run(debug=False)