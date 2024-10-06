import json

# Supongamos que el archivo JSON principal se llama 'artistas.json'
json_file = 'artistas_web.json'

# Este es el texto que contiene las URLs de las imágenes (lo que proporcionaste)
artist_images_text = """
"The Beatles": "https://i.scdn.co/image/ab6761610000e5ebe9348cc01ff5d55971b22433",
  "Bob Marley": "https://i.scdn.co/image/b5aae2067db80f694a980e596e7f49618c1206c9",
  "Miles Davis": "https://i.scdn.co/image/423e826b3c1b23930a255d7cbc2daf733f795507",
  "Madonna": "https://i.scdn.co/image/ab6761610000e5eb4b36d28b55620959821f4a5b",
  "Metallica": "https://i.scdn.co/image/ab6761610000e5eb69ca98dd3083f1082d740e44",
  "Bob Dylan": "https://i.scdn.co/image/ab6772690000c46cd7064356b04a156664a37c4f",
  "Beyoncé": "https://i.scdn.co/image/ab6761610000e5eb247f44069c0bd1781df2f785",
  "Johann Sebastian Bach": "https://i.scdn.co/image/a2ec08fe69ecec2748fbc764aede8f1b03ae8f88",
  "Eminem": "https://i.scdn.co/image/ab6761610000e5eb9a398209a4ef3360dce2dec4",
  "David Bowie": "https://i.scdn.co/image/ab6761610000e5ebb78f77c5583ae99472dd4a49",
  "Ella Fitzgerald": "https://i.scdn.co/image/3c1d23141059785c8a863137d176319329dc17ef",
  "Björk": "https://i.scdn.co/image/ab6761610000e5ebaeb5c04fdd8ec6273850d207",
  "Johnny Cash": "https://i.scdn.co/image/ab6761610000e5eb94a8326675bfcafb20f0a235",
  "Prince": "https://i.scdn.co/image/ab6761610000e5ebeaca358712b3fe4ed9814640",
  "Radiohead": "https://i.scdn.co/image/ab6761610000e5eba03696716c9ee605006047fd",
  "Stevie Wonder": "https://i.scdn.co/image/c59faacbed7aa770266bad048660810eca204108",
  "Nina Simone": "https://i.scdn.co/image/ab6761610000e5eb136c51c848c26a6cce7f9e56",
  "Kraftwerk": "https://i.scdn.co/image/ab6761610000e5ebc46999e5afaf35c91cbe652b",
  "Bob Marley & The Wailers": "https://i.scdn.co/image/b5aae2067db80f694a980e596e7f49618c1206c9",
  "Queen": "https://i.scdn.co/image/ab6761610000e5eb909b2c4b7c768ee03445cd10",
  "Jay-Z": null,
  "Aretha Franklin": "https://i.scdn.co/image/ab6761610000e5ebf12270128127ba170f90097d",
  "Pink Floyd": "https://i.scdn.co/image/e69f71e2be4b67b82af90fb8e9d805715e0684fa",
  "Rihanna": "https://i.scdn.co/image/ab6761610000e5eb99e4fca7c0b7cb166d915789",
  "Chuck Berry": "https://i.scdn.co/image/d58323616d4f5d22b51e9dfe0ba53aedabe53cd4",
  "Nirvana": "https://i.scdn.co/image/84282c28d851a700132356381fcfbadc67ff498b",
  "Lady Gaga": "https://i.scdn.co/image/ab6761610000e5eb60f57316669a4ba12eb37b94",
  "The Rolling Stones": "https://i.scdn.co/image/ab6761610000e5ebe4cea917b68726aadb4854b8",
  "Kendrick Lamar": "https://i.scdn.co/image/ab6761610000e5eb437b9e2a82505b3d93ff1022",
  "Dolly Parton": "https://i.scdn.co/image/ab6761610000e5eb33c82e34ebf51636457796ff",
  "AC/DC": "https://i.scdn.co/image/ab6761610000e5ebc4c77549095c86acb4e77b37",
  "Coldplay": "https://i.scdn.co/image/ab6761610000e5eb1ba8fc5f5c73e7e9313cc6eb",
  "Whitney Houston": "https://i.scdn.co/image/ab6761610000e5ebcd9f60ab57585bf3b77ecc51",
  "Ludwig van Beethoven": "https://i.scdn.co/image/ab6761610000e5eba636b0b244253f602a629796",
  "The Who": "https://i.scdn.co/image/9cd709cabb4a614b4f1dd9ec256a5f30e21f0150",
  "Drake": "https://i.scdn.co/image/ab6761610000e5eb437b9e2a82505b3d93ff1022",
  "Tina Turner": "https://i.scdn.co/image/ab6761610000e5eb22102415297132237b78829d",
  "Daft Punk": "https://i.scdn.co/image/ab6761610000e5eba7bfd7835b5c1eee0c95fa6e",
  "Kanye West": "https://i.scdn.co/image/ab6761610000e5eb6e835a500e791bf9c27a422a",
  "Led Zeppelin": "https://i.scdn.co/image/207803ce008388d3427a685254f9de6a8f61dc2e",
  "Elton John": "https://i.scdn.co/image/ab6761610000e5eb0a7388b95df960b5c0da8970",
  "Marvin Gaye": "https://i.scdn.co/image/cf79bd3e5c787e2ec152eeb1ea5538b0d4cf1434",
  "Taylor Swift": "https://i.scdn.co/image/ab6761610000e5ebe672b5f553298dcdccb0e676",
  "The Beach Boys": "https://i.scdn.co/image/ab6761610000e5eb10d4a1b12c6ceda12d7cd248",
  "Guns N' Roses": "https://i.scdn.co/image/ab6761610000e5eb50defaf9fc059a1efc541f4c",
  "Adele": "https://i.scdn.co/image/ab6761610000e5eb304a1c66dad43b8ae673d43e",
  "John Coltrane": "https://i.scdn.co/image/ab6761610000e5eb73c7f7505c1af82929ec41df",
  "U2": "https://i.scdn.co/image/ab6761610000e5ebe62be215d2ee31bcd97edaba",
  "The Clash": "https://i.scdn.co/image/ab470b275daa38351810a1eb91d107ebdb821302",
  "Frank Sinatra": "https://i.scdn.co/image/fc4e0f474fb4c4cb83617aa884dc9fd9822d4411",
  "Red Hot Chili Peppers": "https://i.scdn.co/image/ab6761610000e5ebc33cc15260b767ddec982ce8",
  "Shakira": "https://i.scdn.co/image/ab6761610000e5eb44270f00ab4402b3fd91bad1",
  "Black Sabbath": "https://i.scdn.co/image/ab6761610000e5eb4870cd833ebe1092601820c3",
  "Destiny's Child": "https://i.scdn.co/image/ab6761610000e5ebf75e64532704bd6acf0b4e76",
  "Green Day": "https://i.scdn.co/image/ab6761610000e5ebd3fa01fd4baad1158dd2fd8b",
  "Elvis Presley": "https://i.scdn.co/image/ab6761610000e5eb9a93e273380982dff84c0d7c",
  "Santana": "https://i.scdn.co/image/ab6761610000e5eb09882b1b7b33732abd60fc38",
  "Foo Fighters": "https://i.scdn.co/image/ab6761610000e5ebc884df599abc793c116cdf15",
  "Tupac Shakur": "https://i.scdn.co/image/ab6761610000e5eb7f5cc432c9c109248ebec1ac",
  "Fleetwood Mac": "https://i.scdn.co/image/ab6761610000e5ebc8752dd511cda8c31e9daee8",
  "Bruno Mars": "https://i.scdn.co/image/ab6761610000e5ebc36dd9eb55fb0db4911f25dd",
  "The Police": "https://i.scdn.co/image/1f73a61faca2942cd5ea29c2143184db8645f0b3",
  "Céline Dion": "https://i.scdn.co/image/ab6761610000e5ebc3b380448158e7b6e5863cde",
  "Pearl Jam": "https://i.scdn.co/image/ab6761610000e5ebe431fb8ba17cdee73e4ce08a",
  "Buddy Holly": "https://i.scdn.co/image/ff6c6ecd90618f5fb0f8fbf51bf8477586c8843b",
  "The Doors": "https://i.scdn.co/image/ab6761610000e5eb440959e022afc20e819050bd",
  "Alicia Keys": "https://i.scdn.co/image/ab6761610000e5ebebfd16a3bca87c31c1e20576",
  "Run-DMC": null,
  "Oasis": "https://i.scdn.co/image/ab6761610000e5eb0522e98a6f0cf1ddbee9a74f",
  "The Supremes": "https://i.scdn.co/image/5c5dc3f7b4569a7727a63a44513c7a602b72da44",
  "Rush": "https://i.scdn.co/image/ab6761610000e5eb896c4b043fb3063178afd716",
  "Stevie Nicks": "https://i.scdn.co/image/ab6761610000e5ebff4ca8de3c5605e889efab75",
  "Talking Heads": "https://i.scdn.co/image/e4c5b04fce8706c87663357b1f78522a3a5c641b",
  "Lauryn Hill": "https://i.scdn.co/image/ccb108f19352e88d6e69b1093d75a2be5efa1c99",
  "Depeche Mode": "https://i.scdn.co/image/ab6761610000e5ebaff13c9484fdad590ccfb73c",
  "Leonard Cohen": "https://i.scdn.co/image/ab6761610000e5ebd09c1295779cfb71683cee99",
  "Rage Against the Machine": "https://i.scdn.co/image/ab6761610000e5ebda4bd2b213cae330e2a4a901",
  "Dusty Springfield": "https://i.scdn.co/image/8b2bbbe1dde191c637271fab24ff66fda4faf4fa",
  "The Velvet Underground": "https://i.scdn.co/image/d69c2cf10323bf08443c7d122f3a1824a760ab57",
  "Snoop Dogg": "https://i.scdn.co/image/ab6761610000e5eb9a398209a4ef3360dce2dec4",
  "Janis Joplin": "https://i.scdn.co/image/ab6761610000e5ebbf0ee2c099003f6886e4da0c",
  "The Cure": "https://i.scdn.co/image/ab6761610000e5ebdd427c765c409ec02dc3a868",
  "Willie Nelson": "https://i.scdn.co/image/ab6772690000c46ca07b9db029b749adaba3c618",
  "Sonic Youth": "https://i.scdn.co/image/af335c11a73e9311516bb28800be58311410d1be",
  "Donna Summer": "https://i.scdn.co/image/ab6761610000e5eb9e708b5a9b76b98301d35123",
  "Bad Bunny": "https://i.scdn.co/image/ab6761610000e5eb744a4243fb6cc938011a98f4",
  "Genesis": "https://i.scdn.co/image/ab6761610000e5ebdfb2b3c69199bc2dfbf3e5b9",
  "Patti Smith": "https://i.scdn.co/image/20e52d0b168541016f901d9ec6d4f2a0e41f193b",
  "Dua Lipa": "https://i.scdn.co/image/ab6761610000e5eb0c68f6c95232e716f0abee8d",
  "The Smiths": "https://i.scdn.co/image/481b980af463122013e4578c08fb8c5cbfaed1e9",
  "Fela Kuti": "https://i.scdn.co/image/ab6761610000e5ebe8670e361382ef18bbe5b93d",
  "Iron Maiden": "https://i.scdn.co/image/ab6761610000e5eb69ca98dd3083f1082d740e44",
  "Frank Ocean": "https://i.scdn.co/image/ab6761610000e5ebee3123e593174208f9754fab",
  "The Ramones": "https://i.scdn.co/image/ab6761610000e5eb5b9f46a5c5bf8243179d56b0",
  "Billie Holiday": "https://i.scdn.co/image/4efa6b6871a0f672b78b3d16fd5a03c6cd212d58",
  "Massive Attack": "https://i.scdn.co/image/c8bbeedb05f38ae5cb982a7daf4bf7129cca892c",
  "Kate Bush": "https://i.scdn.co/image/ab6761610000e5eb187017724e58e78ee1f5a8e4",
  "Slayer": "https://i.scdn.co/image/8c81130db7b5f933412c4906c30327817f1e1b43",
  "Gloria Estefan": "https://i.scdn.co/image/ab6761610000e5ebcb6ddcef04b60c7078151ac7"
"""

# Convertimos el texto en un diccionario
artist_images = {}
for line in artist_images_text.strip().split("\n"):
    if line.strip():
        artist, url = line.split(": ")
        artist = artist.strip('" ')
        url = url.strip('" ,')
        artist_images[artist] = url

# Leemos el archivo JSON principal
with open(json_file, 'r', encoding='utf-8') as f:
    artistas_data = json.load(f)

# Actualizamos el campo "photo_url" con las URLs correctas
for artist in artistas_data['artistas']:
    artist_name = artist['artist_name']
    if artist_name in artist_images:
        artist['photo_url'] = artist_images[artist_name]

# Guardamos el archivo JSON actualizado
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(artistas_data, f, indent=4)

print("JSON actualizado y guardado correctamente.")
