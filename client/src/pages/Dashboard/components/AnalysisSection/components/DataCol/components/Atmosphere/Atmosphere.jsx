import { Card, Stack, Typography } from "@mui/material"
import { Air } from "@mui/icons-material"
import { useContext } from "react"
import { DashboardContext } from "../../../../../../Dashboard"


const Atmosphere = () => {
    // Gets dashboard data
    const { weather } = useContext(DashboardContext)



    return (
        <Card sx={{ padding: 3, height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "start" }}>
            <Typography variant="h5" color="primary" mb={1}>Atmosphere</Typography>
            <Stack direction="row" gap={1} alignItems="center" justifyContent="end">
                <Air />
                <Typography body1>{weather.wind.speed} m/s</Typography>
            </Stack>
            <Stack direction="row" gap={1} alignItems="center" justifyContent="end">
                <Typography body1>Atm. pressure:</Typography>
                <Typography body1>{weather.main.pressure} hPa</Typography>
            </Stack>
            <Stack direction="row" gap={1} alignItems="center" justifyContent="end">
                <Typography body1>Humidity:</Typography>
                <Typography body1>{weather.main.humidity}%</Typography>
            </Stack>
        </Card>
    )
}

export default Atmosphere