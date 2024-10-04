package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"email-api/mail"

	"github.com/joho/godotenv"
)

func init() {
	err := godotenv.Load()
	if err != nil {
		log.Println("Advertencia: no se cargó el archivo .env, utilizando variables de entorno del sistema.")
	}
}

// Estructura para los artistas
type Artist struct {
	Nombre      string `json:"nombre"`
	FechaEvento string `json:"fecha_evento"`
	HoraEvento  string `json:"hora_evento"`
	Imagen      string `json:"imagen"`
}

// Estructura para la solicitud de ticket
type TicketRequest struct {
	Artist  Artist `json:"artist"`
	QRCode  string `json:"qr_code"` // Ruta a la imagen del código QR
	Email   string `json:"email"`   // Correo electrónico del destinatario
}

// Estructura para recibir los datos del correo de recomendación
type EmailRequest struct {
	Mail            string   `json:"mail"`
	Subject         string   `json:"subject"`
	Body            string   `json:"body"`
	Artists         []Artist `json:"artists"`
	DestinationEmail string  `json:"destinationEmail"` // Dirección del destinatario
}

// Habilitar CORS
func enableCors(w *http.ResponseWriter) {
	(*w).Header().Set("Access-Control-Allow-Origin", "*")
	(*w).Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	(*w).Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

// Generar la sección de artistas en HTML para el correo de recomendación
func generateArtistsSection(artists []Artist) string {
	var artistsHTML string
	for _, artist := range artists {
		artistsHTML += fmt.Sprintf(`
			<tr>
				<td style="background-color: #ffffff; padding: 20px 30px;">
					<table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%%">
						<tr>
							<td style="width: 150px;">
								<img src="%s" alt="%s" width="150" height="150" style="border-radius: 10px;">
							</td>
							<td style="padding-left: 20px; vertical-align: top;">
								<h3 style="margin: 0; font-family: sans-serif; font-size: 18px; line-height: 24px; color: #333333;">%s</h3>
								<p style="margin: 5px 0 0 0; font-family: sans-serif; font-size: 14px; line-height: 20px; color: #555555;">Fecha: %s</p>
								<p style="margin: 0; font-family: sans-serif; font-size: 14px; line-height: 20px; color: #555555;">Hora: %s</p>
							</td>
						</tr>
					</table>
				</td>
			</tr>
		`, artist.Imagen, artist.Nombre, artist.Nombre, artist.FechaEvento, artist.HoraEvento)
	}
	return artistsHTML
}

// Handler para enviar el correo de recomendación (endpoint /recommend)
func sendEmailHandler(w http.ResponseWriter, r *http.Request) {
	enableCors(&w) // Habilitar CORS para todas las solicitudes

	// Manejar las solicitudes OPTIONS para CORS
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Asegurarse de que sea un POST
	if r.Method != http.MethodPost {
		http.Error(w, "Método no permitido", http.StatusMethodNotAllowed)
		return
	}

	// Decodificar el JSON de la solicitud
	var emailReq EmailRequest
	err := json.NewDecoder(r.Body).Decode(&emailReq)
	if err != nil {
		http.Error(w, "Error al procesar el JSON", http.StatusBadRequest)
		return
	}

	// Generar la sección de artistas en HTML
	artistsSection := generateArtistsSection(emailReq.Artists)

	// Construir el contenido del correo usando la plantilla
	content := fmt.Sprintf(`
		<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
		<html xmlns="http://www.w3.org/1999/xhtml" lang="es">
		<head>
			<meta charset="UTF-8">
			<meta name="viewport" content="width=device-width, initial-scale=1.0">
			<title>¡Descubre Tu Próximo Concierto Inolvidable!</title>
			<style type="text/css">
				body, table, td, a { -webkit-text-size-adjust: 100%%; -ms-text-size-adjust: 100%%; }
				table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
				img { -ms-interpolation-mode: bicubic; }
				body { height: 100%% !important; margin: 0 !important; padding: 0 !important; width: 100%% !important; }
				img { border: 0; height: auto; line-height: 100%%; outline: none; text-decoration: none; }
				table { border-collapse: collapse !important; }
				.email-container { width: 100%% !important; }
				.fluid { max-width: 100%% !important; height: auto !important; margin-left: auto !important; margin-right: auto !important; }
				.stack-column { display: block !important; width: 100%% !important; max-width: 100%% !important; direction: ltr !important; }
				.center-on-narrow { text-align: center !important; display: block !important; margin-left: auto !important; margin-right: auto !important; }
			</style>
		</head>
		<body>
			<center>
				<div style="max-width: 600px; margin: 0 auto;" class="email-container">
					<!-- Header -->
					<table align="center" cellspacing="0" cellpadding="0" border="0" width="100%%" style="margin: auto;">
						<tr>
							<td style="padding: 20px; text-align: center; background-color: #6a11cb;">
								<h1 style="margin: 0; font-family: sans-serif; font-size: 28px; color: #ffffff; font-weight: bold;">Descubre Tu Música</h1>
								<p style="margin: 0; font-family: sans-serif; font-size: 16px; color: #ffffff;">Experiencias sonoras que transforman vidas</p>
							</td>
						</tr>
					</table>
					<!-- Contenido principal -->
					<table align="center" cellspacing="0" cellpadding="0" border="0" width="100%%" style="margin: auto;">
						<tr>
							<td style="padding: 40px 30px;">
								<h2 style="margin: 0 0 10px 0; font-family: sans-serif; font-size: 22px;">¡Hola, %s!</h2>
								<p style="margin: 0;">Hemos seleccionado cuidadosamente algunos artistas que creemos encajarán perfectamente con tus gustos musicales. ¡No te pierdas la oportunidad de vivir estas experiencias únicas en vivo!</p>
							</td>
						</tr>
						<tr>
							<td style="background-color: #ffffff;">
								<h2 style="margin: 0; padding: 0 30px 20px; font-family: sans-serif; font-size: 24px; text-align: center;">Conciertos Recomendados</h2>
							</td>
						</tr>
						<!-- Sección de Artistas -->
						%s
						<!-- CTA -->
						<tr>
							<td style="padding: 0 30px 40px;">
								<table align="center" cellspacing="0" cellpadding="0" border="0" style="margin: auto;">
									<tr>
										<td style="background: #6a11cb; text-align: center;">
											<a href="#" style="background: #6a11cb; border: 15px solid #6a11cb; font-family: sans-serif; font-size: 16px; color: #ffffff; text-decoration: none; display: block;">
												<strong>Explorar Más Conciertos</strong>
											</a>
										</td>
									</tr>
								</table>
							</td>
						</tr>
					</table>
				</div>
			</center>
		</body>
		</html>
	`, emailReq.Mail, artistsSection)

	// Crear el remitente usando el paquete mail
	sender := mail.NewGmailSender(
		os.Getenv("EMAIL_SENDER_NAME"),
		os.Getenv("EMAIL_SENDER_ADDRESS"),
		os.Getenv("EMAIL_SENDER_PASSWORD"),
	)

	// Enviar el correo
	to := []string{emailReq.DestinationEmail}
	attachFiles := []string{} // Puedes agregar archivos si es necesario

	err = sender.SendEmail(emailReq.Subject, content, to, nil, nil, attachFiles)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error al enviar el correo: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Correo enviado exitosamente"))
}

// Generar el ticket HTML basado en el template (endpoint /ticket)
func generateTicketHTML(ticketReq TicketRequest) string {
	htmlTemplate := `
	<html xmlns="http://www.w3.org/1999/xhtml" lang="es">
	<head>
	    <meta charset="UTF-8">
	    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	    <title>¡Tu Entrada para el Concierto!</title>
	    <style type="text/css">
	        body, table, td, a { -webkit-text-size-adjust: 100%%; -ms-text-size-adjust: 100%%; }
	        table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
	        img { -ms-interpolation-mode: bicubic; }
	        body { margin: 0; padding: 0; width: 100%%; background-color: #F8F9FA; }
	        img { border: 0; height: auto; outline: none; text-decoration: none; }
	        table { border-collapse: collapse !important; }
	        .ticket-container { max-width: 600px; margin: 20px auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
	        .artist-photo { width: 150px; height: 150px; border-radius: 10px; }
	        .qr-code { width: 150px; height: 150px; margin-top: 20px; }
	        h2 { font-family: sans-serif; font-size: 24px; color: #333; }
	        p { font-family: sans-serif; font-size: 16px; color: #555; }
	    </style>
	</head>
	<body>
	    <center>
	        <div class="ticket-container">
	            <table width="100%%" cellpadding="0" cellspacing="0">
	                <tr>
	                    <td align="center">
	                        <h2>¡Aquí está tu entrada para el evento que tanto esperabas!</h2>
	                    </td>
	                </tr>
	                <tr>
	                    <td align="center">
	                        <h3>Artista: %s</h3>
	                        <p>Fecha del evento: %s</p>
	                        <p>Hora del evento: %s</p>
	                    </td>
	                </tr>
	                <tr>
	                    <td align="center">
	                        <img src="%s" alt="Foto de %s" class="artist-photo">
	                    </td>
	                </tr>
	                <tr>
	                    <td align="center">
	                        <img src="%s" alt="Código QR" class="qr-code">
	                    </td>
	                </tr>
	            </table>
	        </div>
	    </center>
	</body>
	</html>
	`
	return fmt.Sprintf(htmlTemplate,
		ticketReq.Artist.Nombre, ticketReq.Artist.FechaEvento, ticketReq.Artist.HoraEvento,
		ticketReq.Artist.Imagen, ticketReq.Artist.Nombre,
		ticketReq.QRCode)
}

// Enviar correo electrónico con el ticket
func sendTicketByEmail(to string, subject string, body string) error {
	// Usamos el paquete mail para crear el remitente
	sender := mail.NewGmailSender(
		os.Getenv("EMAIL_SENDER_NAME"),
		os.Getenv("EMAIL_SENDER_ADDRESS"),
		os.Getenv("EMAIL_SENDER_PASSWORD"),
	)

	// Enviar el correo con el ticket
	toAddresses := []string{to}
	attachFiles := []string{} // Puedes agregar archivos si es necesario
	return sender.SendEmail(subject, body, toAddresses, nil, nil, attachFiles)
}

// Handler para generar el ticket y enviarlo por correo (endpoint /ticket)
func ticketHandler(w http.ResponseWriter, r *http.Request) {
	enableCors(&w) // Habilitar CORS para todas las solicitudes

	// Asegurarse de que sea un POST
	if r.Method != http.MethodPost {
		http.Error(w, "Método no permitido", http.StatusMethodNotAllowed)
		return
	}

	// Decodificar el JSON de la solicitud
	var ticketReq TicketRequest
	err := json.NewDecoder(r.Body).Decode(&ticketReq)
	if err != nil {
		http.Error(w, "Error al procesar el JSON", http.StatusBadRequest)
		return
	}

	// Generar el HTML del ticket
	ticketHTML := generateTicketHTML(ticketReq)

	// Enviar el ticket por correo electrónico
	err = sendTicketByEmail(ticketReq.Email, "Tu entrada para el concierto", ticketHTML)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error al enviar el correo: %v", err), http.StatusInternalServerError)
		return
	}

	// Respuesta de éxito
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Correo enviado exitosamente con el ticket"))
}

func main() {
	// Endpoint para enviar correo de recomendación
	http.HandleFunc("/recommend", sendEmailHandler)

	// Endpoint para generar y enviar ticket
	http.HandleFunc("/ticket", ticketHandler)

	fmt.Println("Servidor escuchando en el puerto 8080...")
	log.Fatal(http.ListenAndServe(":8080", nil))
}



/* {
	"mail": "Lexo",
	"subject": "Descubre tu próximo concierto",
	"body": "Basado en tus gustos te mostramos los mejores conciertos.",
	"artists": [
	  {
		"nombre": "Artista 1",
		"fecha_evento": "2024-10-15",
		"hora_evento": "20:00",
		"imagen": "https://link_a_la_imagen.com/artista1.jpg"
	  },
	  {
		"nombre": "Artista 2",
		"fecha_evento": "2024-11-05",
		"hora_evento": "21:00",
		"imagen": "https://link_a_la_imagen.com/artista2.jpg"
	  }
	],
	"destinationEmail": "lucasabello4@gmail.com"
  } */
  
/*   {
  "artist": {
    "nombre": "Artista 1",
    "fecha_evento": "2024-10-15",
    "hora_evento": "20:00",
    "imagen": "https://link_a_la_imagen.com/artista1.jpg"
  },
  "qr_code": "https://link_a_la_imagen.com/codigo_qr.jpg",
  "email": "lucasabello4@gmail.com"
}
 */
  