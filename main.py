from flask import Flask, Response, make_response, jsonify, request, url_for, redirect, render_template
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load environment variables (production values are provided via app.yaml)
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')
app.config['BASE_URL'] = os.getenv('BASE_URL', 'https://ta-customrss.as.r.appspot.com')
app.config['PREFERRED_URL_SCHEME'] = 'https'

BASE_URL = "https://ta-customrss.as.r.appspot.com"

mongo_uri = os.getenv('MONGO_URI')
conn = MongoClient(mongo_uri)
db = conn.get_database('test_data_berita')
users_collection = db.test_users2

def default_news_data():
    collection_1 = db.s_data_berita
    data = list(collection_1.find().sort("_id", -1).limit(500)) 
    dfdata = pd.DataFrame(data)
    dfdata['_id'] = dfdata['_id'].astype(str)
    dfdata["pub_date"] = pd.to_datetime(dfdata["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")
    dfdata = dfdata.sort_values(by="pub_date", ascending=False)
    dfdata = dfdata.reset_index(drop=True)

    return dfdata

def recommended_news_data():
    users = users_collection.find()

    recs_df = []
    for user in users:
        user_id = user["user_id"]

        # Check if the user exists in the database
        user = users_collection.find_one({"user_id": user_id}, {"recommendations": 1})

        recommended_news_dict = user["recommendations"]

        if len(recommended_news_dict) > 0:
            recs_df.append(recommended_news_dict)

    flat_recs_list = [item for sublist in recs_df for item in sublist]

    combined_recs_df = pd.DataFrame(flat_recs_list)

    return combined_recs_df

def rfc822_date():
        now = datetime.utcnow()
        return now.strftime("%a, %d %b %Y %H:%M:%S GMT")

def generate_recs_time(df1, rec_length):
    t_rec_length = rec_length + 1
    # Get the last time from df1 and latest from df2
    last_time_1 = df1["pub_date"][4]
    latest_time_2 = df1["pub_date"][5]

    interval = (latest_time_2 - last_time_1) / t_rec_length

    timestamps = [last_time_1 + i * interval for i in range(1,t_rec_length)]

    return timestamps

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

def generate_feed(user_id, filtered_df, sort_flag):
    all_news_df = default_news_data()
    
    news = filtered_df.to_dict(orient="records")
    
    remaining_df = all_news_df[~all_news_df['_id'].isin(filtered_df['_id'])].copy()
    remaining_news = remaining_df.to_dict(orient="records")

    if sort_flag:
        new_timestamps = generate_recs_time(all_news_df, len(news))
        for n, t in zip(news, new_timestamps):
            n['pub_date'] = t

    news.extend(remaining_news)
    
    return generate_default_rss_feed(user_id, news)

def all_news_data():
    collection_1 = db.s_data_berita
    data = list(collection_1.find())
    dfdata = pd.DataFrame(data)
    dfdata['_id'] = dfdata['_id'].astype(str)
    dfdata["pub_date"] = pd.to_datetime(dfdata["pub_date"], format="%a, %d %b %Y %H:%M:%S %z")
    dfdata = dfdata.sort_values(by="pub_date", ascending=False)
    dfdata = dfdata.reset_index(drop=True)

    return dfdata

def generate_unique_recommendations(user_id):
    clicked_news = users_collection.find_one({"user_id": user_id}, {"clicked_news": 1})

    # Sort clicked news by ascending time and ensure they are unique
    sorted_news_ids = []
    if clicked_news and "clicked_news" in clicked_news:
        sorted_news_ids = [
            news["news_id"] for news in sorted(clicked_news["clicked_news"], key=lambda x: x["timestamp"])
        ]
    unique_news_ids = []
    for item in sorted_news_ids:
        if item not in unique_news_ids:
            unique_news_ids.append(item)

    # Initialize data and recommendation number
    df1 = all_news_data()
    rec_num = 5

    # TfidfVectorizer by default
    tfidf_matrix = TfidfVectorizer()

    # Get document (all news) and query (clicked news) values
    cleanedt = df1['Cleaned_sw_title'].values
    filtered_df = df1[df1['_id'].isin(unique_news_ids)]
    c_indexes = filtered_df.index.tolist()
    titles = filtered_df['Cleaned_sw_title'].tolist()

    # Fit document to vectorizer and transform
    tfidf_matrix.fit_transform(cleanedt)
    tfidf_vector = tfidf_matrix.transform(cleanedt)

    all_recommended_articles = []

    # Recommendation iteration
    for i, doc in enumerate(titles):
        i_tfidf_vector = tfidf_matrix.transform([doc])
        similarity = cosine_similarity(i_tfidf_vector, tfidf_vector)

        # Rank articles by similarity, excluding itself
        rank = similarity.flatten().argsort()[::-1]
        recommended_indices = [idx for idx in rank if idx not in c_indexes][:rec_num]

        all_recommended_articles.extend(recommended_indices)

    # Get article IDs
    filtereddf = df1.iloc[all_recommended_articles]
    filtereddf = filtereddf.drop_duplicates(subset=['_id'])

    return filtereddf.to_dict(orient="records")

@app.route("/updaterecs", methods=["GET"])
def update_rss_feed():
    users = users_collection.find()
        
    recs_log = []
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
            recommended_news = generate_unique_recommendations(user_id)
        else:
            # Default to including all news items in no specific order
            recommended_news = []

        rec_time = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
            
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"recommendations": recommended_news,
                    "last_recommendation_time": rec_time}
            }
        )

        recs_log.append(f"Sebanyak {len(recommended_news)} rekomendasi dihasilkan untuk user {user_id}")
    
    return jsonify({recs_log})

@app.route("/")
def welcome():
    # Generate a new user ID
    user_id = str(uuid.uuid4())

    # Fetch all news data and sort in descending order
    all_news_df = default_news_data()

    remaining_news = all_news_df.to_dict(orient="records")

    pretty_rss_feed = generate_default_rss_feed(user_id, remaining_news)
    
    # Store the user in the database
    users_collection.insert_one({
        "user_id": user_id,
        "created_at": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "clicked_news": [],
        "last_recommendation_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "recommendations": [],
        "rss_feed_sorted": pretty_rss_feed,
        "rss_feed_unsorted": pretty_rss_feed
    })
    
    # Generate the RSS feed URL for the user
    rss_url = url_for("generate_rss_feed", user_id=user_id, _external=True)
   
    # Render the welcome page with the custom RSS URL
    return render_template(
        'welcomepg2.html',
        base_rss_url = rss_url
    )

@app.route("/updaterss", methods=["GET"])
def update_rss_feed():
    users = users_collection.find()
    
    with app.app_context():
        for user in users:
            user_id = user["user_id"]
            
            # Check if the user exists in the database
            user = users_collection.find_one({"user_id": user_id}, {"recommendations": 1})
            if not user:
                return "Invalid user ID", 404

            recommended_news_dict = user["recommendations"]
            
            if len(recommended_news_dict) > 0:
                recommended_news_df = pd.DataFrame(recommended_news_dict)      

                # Generate both feeds
                rss_sorted = generate_feed(user_id, recommended_news_df, sort_flag=True)
                rss_unsorted = generate_feed(user_id, recommended_news_df, sort_flag=False)
            else:
                all_news_df = default_news_data()
                remaining_news = all_news_df.to_dict(orient="records")

                rss_sorted = generate_default_rss_feed(user_id, remaining_news)
                rss_unsorted = generate_default_rss_feed(user_id, remaining_news)
                
            rec_time = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
                
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"recommendations": recommended_news_dict,
                        "last_recommendation_time": rec_time,
                        "rss_feed_sorted": rss_sorted,
                        "rss_feed_unsorted": rss_unsorted}
                }
            )
        
        return jsonify({"message": "RSS feed updated for all users"})

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

    if sort_flag:
        pretty_rss_feed = user["rss_feed_sorted"]
    else:
        pretty_rss_feed = user["rss_feed_unsorted"]

    # Return the RSS feed content
    response = make_response(pretty_rss_feed)
    response.headers['Content-Type'] = 'application/rss+xml'
    response.headers['Last-Modified'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    return response

@app.route('/uat_recommendations', methods=['GET'])
def get_uat_recommendations():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID not found."})

    # Fetch recommendations from MongoDB
    user_data = users_collection.find_one({'user_id': user_id})

    # Process clicked news
    if "clicked_news" in user_data:
        clicked_titles = [
            news["clicked_title"] for news in sorted(user_data["clicked_news"], key=lambda x: x["timestamp"])
        ]
    else:
        clicked_titles = []
  
    # Process recommendations
    if "recommendations" in user_data:
        recs_titles = [
            news["Title"] for news in user_data["recommendations"]
        ]
    else:
        recs_titles = []

    return jsonify({
        "clicked_news": clicked_titles,
        "recommended_news": recs_titles
    })

@app.route("/track", methods=["GET"])
def track():
    raw_query_string = request.query_string.decode("utf-8")
    query_params = parse_qs(unescape(raw_query_string))  # Decode and parse

    user_id = query_params.get("user_id", [None])[0]
    news_id = query_params.get("news_id", [None])[0]

    redirect_to = request.args.get("redirect", "true")  # Extract 'redirect' parameter
    
    dfdata = default_news_data()
    combined_recs_df = recommended_news_data()
    
    dfs = [dfdata, combined_recs_df]
    news_data_concat = pd.concat(dfs)
    news_data = news_data_concat.to_dict(orient="records")

    # Validate parameters
    if redirect_to == "true" and news_id is not None:
        news_item = next((item for item in news_data if str(item["_id"]) == str(news_id)), None)
        if not news_item:
            return "News item not found", 404

        original_url = news_item["Link"]
        original_title = news_item["Title"]

        # Log the click
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")        
        
        users_collection.update_one(
        {"user_id": user_id},
        {"$push": {"clicked_news": {
                    "news_id": news_id,
                    "clicked_title": original_title,
                    "timestamp": timestamp}
                    }
        })

        return redirect(original_url)
    
    return "Invalid tracking parameters", 400

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_rss_feed, 'interval', minutes=50)
    scheduler.start()

    app.run(debug=False)