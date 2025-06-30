import { Stack } from "@mui/material";
import { GoogleLogin } from "@react-oauth/google";
import { useContext } from "react";
import { DataContext } from "../../context/DataContext";

const GoogleButton = ({ setError, text = "signin_with" }) => {
    // Gets global data from the context
    const { crud, setAccess, setRefresh, navigate, setLoading } = useContext(DataContext)



    // Google authentication logic
    const handleGoogleLoginSuccess = async (credentialResponse) => {
        setLoading(true)

        const token = credentialResponse.credential;

        const response = await crud({
            method: 'post',
            url: '/users/google-login/',
            body: {
                token
            },
        });

        console.log(response)

        if (response.status === 200) {
            localStorage.setItem('access', response.data.access);
            setAccess(response.data.access);
            localStorage.setItem('refresh', response.data.refresh);
            setRefresh(response.data.refresh);
            navigate('/dashboard');
        } else {
            setError('Google register failed.');
        }

        setLoading(false)
    };

    const handleGoogleLoginFailure = () => {
        setError('Google register failed.');
    };



    return (
        <Stack alignItems="center">
            <GoogleLogin
                onSuccess={handleGoogleLoginSuccess}
                onError={handleGoogleLoginFailure}
                size="large"
                theme="filled_black"
                logo_alignment="center"
                shape="pill"
                text={text}
            />
        </Stack>
    )
}

export default GoogleButton