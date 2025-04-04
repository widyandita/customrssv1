from flask import Flask, Response, jsonify, request, url_for, redirect, session, render_template
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
#app.secret_key = "testingupdaterssdita"
app.config['SERVER_NAME'] = '127.0.0.1:5000'
app.config['APPLICATION_ROOT'] = '/'        
app.config['PREFERRED_URL_SCHEME'] = 'http'

conn = MongoClient('mongodb://localhost:27017')
db = conn.get_database('test_data_berita')
users_collection = db.test_users2

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

def generate_unique_recommendations(user_id):
    #user_id = session.get("user_id")
    #print(user_id)
    # Retrieve clicked news data from MongoDB, sorted by timestamp
    clicked_news = users_collection.find_one({"user_id": user_id}, {"clicked_news": 1})

    if clicked_news and "clicked_news" in clicked_news:
        # Sort the clicked_news by timestamp and extract the news_id
        sorted_news_ids = [
            news["news_id"] for news in sorted(clicked_news["clicked_news"], key=lambda x: x["timestamp"])
        ]
    #clicked_news2 = list(reversed(clicked_news))
    # print(clicked_news)
    # print(clicked_news2)
    
    # Ensure there are at least 5 unique clicked news IDs
    unique_news_ids = []
    for item in sorted_news_ids:
        if item not in unique_news_ids:
            unique_news_ids.append(item)
        if len(unique_news_ids) >= 5:
            break
    
    if len(unique_news_ids) < 5:
        # print(clicked_news)
        # print(unique_news_ids)
        print("Not enough unique clicked news IDs for recommendation generation.")
        return
    
    # Generate unique recommendations
    unique_recommendations = set()
    for news_id in unique_news_ids[:5]:
        recs = generate_recommendations(news_id)
        for rec in recs:
            unique_recommendations.add(rec)
    
    # Print or save the recommendations to the database
    recommendations_list = list(unique_recommendations)
    rec_time = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"recommendations": recommendations_list,
                  "last_recommendation_time": rec_time}
        }
    )
    #print("Generated Recommendations:", recommendations_list)
    #print(unique_recommendations)

    return recommendations_list

@app.route("/")
def welcome():
    #session.clear()
    # Generate a new user ID
    user_id = str(uuid.uuid4())
    
    # Store the user in the database
    users_collection.insert_one({
        "user_id": user_id,
        "created_at": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "clicked_news": [],
        "last_recommendation_time": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "recommendations": []
    })

    #session["user_id"] = user_id
    
    # Generate the RSS feed URL for the user
    rss_url = url_for("generate_rss_feed", user_id=user_id, _external=True)
    
    # Render the welcome page with the custom RSS URL
    return render_template(
        'welcomepg.html',
        rss_url = rss_url
    )

@app.route("/updaterss", methods=["GET"])
def update_rss_feed():
    def rfc822_date():
        now = datetime.utcnow()
        return now.strftime("%a, %d %b %Y %H:%M:%S GMT")
        # rec_time = users_collection.find_one({"user_id": user_id}, {"last_recommendation_time": 1})
        # rec_time = rec_time['last_recommendation_time']

        # return rec_time
    #user_id = request.args.get('user_id')
    #session["user_id"] = user_id
    BASE_URL = "http://127.0.0.1:5000"
    users = users_collection.find()
    with app.app_context():
        for user in users:
            user_id = user["user_id"]
            
            #user_id = session.get("user_id")
            
            # Check if the user exists in the database
            user = users_collection.find_one({"user_id": user_id}, {"clicked_news": 1})
            if not user:
                return "Invalid user ID", 404

            #tracked_news_id = session.get("tracked_news_id", None)

            # print(user_id)
            # print(tracked_news_id)
            if user and "clicked_news" in user:
            # Sort the clicked_news by timestamp and extract the news_id
                sorted_news_ids = [
                    news["news_id"] for news in sorted(user["clicked_news"], key=lambda x: x["timestamp"])
                ]

            if len(sorted_news_ids) > 0:
                # Generate recommendations based on the tracked ID
                recommended_ids = generate_unique_recommendations(user_id)
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
            description.text = "Kumpulan berita terbaru dari berbagai situs berita Indonesia. Silahkan memilih minimal 5 berita untuk mendapatkan rekomendasi berita."
            last_build_date = SubElement(channel, "lastBuildDate")
            last_build_date.text = rfc822_date()

            # Add items in the correct order
            for news in ordered_news:
                item = SubElement(channel, "item")
                item_title = SubElement(item, "title")
                item_title.text = news["Title"]
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

            #session.pop("tracked_news_id", None)

            # print(ordered_news)  # Debugging output for order verification
            # print(tracked_news_id)

            #print(recommended_ids)
            users_collection.update_one({"user_id": user_id}, {"$set": {"rss_feed": pretty_rss_feed}})
    
        return jsonify({"message": "RSS feeds updated for all users"})

@app.route("/rss", methods=["GET"])
def generate_rss_feed():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Fetch the user's RSS feed from the database
    user = users_collection.find_one({"user_id": user_id})
    if not user or "rss_feed" not in user:
        return jsonify({"error": "RSS feed not found for the user"}), 404
    
    # Return the RSS feed content
    response = Response(user["rss_feed"], content_type="application/rss+xml")
    return response

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
        #collection_2 = db.test_log3
        users_collection.update_one(
        {"user_id": user_id},
        {"$push": {"clicked_news": {
            "news_id": news_id, "timestamp": timestamp
        }}}
        )
              
        #session["tracked_news_id"] = news_id
        # print(f"Tracked news ID saved in session: {news_id}")

        return redirect(original_url)
    
    return "Invalid tracking parameters", 400

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_rss_feed, 'interval', minutes=5)
    scheduler.start()

    app.run(debug=True)