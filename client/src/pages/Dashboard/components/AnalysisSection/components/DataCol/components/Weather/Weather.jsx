import { Box, Card, Stack, Typography } from "@mui/material"
import { useContext } from "react"
import { DashboardContext } from "../../../../../../Dashboard"


const Weather = () => {
    // Gets dashboard data
    const { weather } = useContext(DashboardContext)



    const kelvinToCels = (t) => { return Math.floor(t - 273.15) }


    return (
        <Card sx={{ padding: 2, height: "100%", display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <Stack direction="row" alignItems="center">
                <img className="weather-icon" src={`https://openweathermap.org/img/wn/${weather.weather[0].icon}@2x.png`} alt="" />
                <Box>
                    <Typography color="primary" variant="body1">{kelvinToCels(weather.main.temp)}°C</Typography>
                    <Typography variant="body2" color="text.secondary">{weather.weather[0].main}</Typography>
                </Box>
            </Stack>
            <Typography variant="body2" color="primary.dark" textAlign="center">Feels like {kelvinToCels(weather.main.feels_like)}°C</Typography>
            <Typography variant="body2" textAlign="center">{kelvinToCels(weather.main.temp_min)}°C - {kelvinToCels(weather.main.temp_max)}°C</Typography>
        </Card>
    )
}

export default Weather