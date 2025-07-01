import { Box, Card, Stack, Typography } from "@mui/material"
import { useContext } from "react"
import { DashboardContext } from "../../../../../../Dashboard"


const Weather = () => {
    // Gets dashboard data
    const { weather } = useContext(DashboardContext)



    const kelvinToCels = (t) => { return Math.floor(t - 273.15) }


    return (
        <Card sx={{ 
            padding: 2,
            height: "100%",
            display: "flex",
            flexDirection: { xs: "row", sm: "column", md: "row", lg: "column" },
            justifyContent: { xs: "start", sm: "center", md: "start", lg: "center" },
            gap: { xs: 2, sm: 0, md: 2, lg: 0 },
            alignItems: { xs: "center", sm: "start", md: "center", lg: "start" }
        }}>
            <Stack direction="row" width={"100%"} alignItems="center" justifyContent={{ xs: "start", sm: "center", md: "start" }}>
                <img className="weather-icon" src={`https://openweathermap.org/img/wn/${weather.weather[0].icon}@2x.png`} alt="" />
                <Box>
                    <Typography color="primary" variant="body1">{kelvinToCels(weather.main.temp)}°C</Typography>
                    <Typography variant="body2" color="text.secondary">{weather.weather[0].main}</Typography>
                </Box>
            </Stack>
            <Stack width={"100%"}>
                <Typography variant="body2" color="primary.dark" textAlign={{ xs: "start", sm: "center", md: "start", lg: "center" }}>Feels like {kelvinToCels(weather.main.feels_like)}°C</Typography>
                <Typography variant="body2" textAlign={{ xs: "start", sm: "center", md: "start", lg: "center" }}>{kelvinToCels(weather.main.temp_min)}°C - {kelvinToCels(weather.main.temp_max)}°C</Typography>
            </Stack>
        </Card>
    )
}

export default Weather