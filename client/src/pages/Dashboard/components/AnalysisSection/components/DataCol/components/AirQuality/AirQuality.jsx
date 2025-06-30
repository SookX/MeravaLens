import { Card, Grid, Stack, Typography } from "@mui/material"
import AirComponent from "./components/AirComponent/AirComponent"
import { useContext } from "react"
import { DashboardContext } from "../../../../../../Dashboard"


const AirQuality = () => {
    // Gets dashboard data
    const { airPollution } = useContext(DashboardContext)



    const airQual = (aqi) => {
        switch(aqi) {
            case 1:
                return {
                    label: "Good",
                    color: "success"
                }
            case 2:
                return {
                    label: "Fair",
                    color: "success"
                }
            case 3:
                return {
                    label: "Moderate",
                    color: "warning"
                }
            case 4:
                return {
                    label: "Poor",
                    color: "error"
                }
            case 5:
                return {
                    label: "Very Poor",
                    color: "error"
                }
        }
    }



    return (
        <Card sx={{ padding: 3 }}>
            <Typography variant="h4" color="primary">Air Quality</Typography>
            <Stack direction="row" mb={2} gap={1}>
                <Typography variant="body1">The air quality in the region is:</Typography>
                <Typography variant="body1" color={airQual(airPollution.list[0].main.aqi).color}>{airQual(airPollution.list[0].main.aqi).label}</Typography>

            </Stack>
            <Typography color="primary.dark" mb={1} variant="body1">Component distribution (in μg/m3):</Typography>

            <Grid container>
                <AirComponent label="CO" objKey="co" />
                <AirComponent label="NO" objKey="no" />
                <AirComponent label="NO2" objKey="no2" />
                <AirComponent label="O3" objKey="o3" />
                <AirComponent label="SO2" objKey="so2" />
                <AirComponent label="PM2.5" objKey="pm2_5" />
                <AirComponent label="PM10" objKey="pm10" />
                <AirComponent label="NH3" objKey="nh3" />
            </Grid>
        </Card>
    )
}

export default AirQuality