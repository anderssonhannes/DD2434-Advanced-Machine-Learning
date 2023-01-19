import requests

url = "https://wft-geo-db.p.rapidapi.com/v1/geo/cities"

querystring = {"minPopulation":"100000"}

headers = {
	"X-RapidAPI-Key": "2632472b80msh616908f46d08403p17a3b2jsn86eb032a3236",
	"X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)