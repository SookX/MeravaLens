import { Grid } from "@mui/material"
import Location from "./components/Location/Location"
import Weather from "./components/Weather/Weather"
import Atmosphere from "./components/Atmosphere/Atmosphere"
import AirQuality from "./components/AirQuality/AirQuality"


const DataCol = ({ airPollution }) => {
    return (
        <Grid container spacing={2}>
            <Grid size={12}>
                <Location />
            </Grid>
            <Grid size={5}>
                <Weather />
            </Grid>
            <Grid size={"grow"}>
                <Atmosphere />
            </Grid>
            <Grid size={12}>
                <AirQuality />
            </Grid>
        </Grid>
    )
}

export default DataCol